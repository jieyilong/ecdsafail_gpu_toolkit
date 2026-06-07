// CUDA secp256k1 island searcher — Step 1: field arithmetic + EC, self-validated.
// u256 = uint32_t[8], little-endian. p = 2^256 - 2^32 - 977.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

typedef uint32_t u32;
typedef uint64_t u64;

struct U256 { u32 v[8]; };

// p and c=2^256-p=2^32+977
__device__ __constant__ u32 P[8]  = {0xFFFFFC2F,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF};
// c = 0x1000003D1
#define C_LOW  0x000003D1u   // 977
// c = 2^32 + 977 -> as a value: limb0=0x000003D1, limb1=0x1 ... we handle c-mul specially

__device__ __forceinline__ bool geP(const u32 a[8]){
    for(int i=7;i>=0;i--){ if(a[i]!=P[i]) return a[i]>P[i]; }
    return true; // equal -> >=
}
__device__ __forceinline__ void subP(u32 a[8]){
    u64 b=0;
    for(int i=0;i<8;i++){ u64 t=(u64)a[i]-P[i]-b; a[i]=(u32)t; b=(t>>63)&1; }
}
// a = a + b mod p (a,b < p)
__device__ __forceinline__ void addmod(u32 a[8], const u32 b[8]){
    u64 c=0; for(int i=0;i<8;i++){ u64 t=(u64)a[i]+b[i]+c; a[i]=(u32)t; c=t>>32; }
    if(c || geP(a)) subP(a);
}
// a = a - b mod p
__device__ __forceinline__ void submod(u32 a[8], const u32 b[8]){
    u64 br=0; u32 t[8];
    for(int i=0;i<8;i++){ u64 d=(u64)a[i]-b[i]-br; t[i]=(u32)d; br=(d>>63)&1; }
    if(br){ // add p
        u64 c=0; for(int i=0;i<8;i++){ u64 s=(u64)t[i]+P[i]+c; t[i]=(u32)s; c=s>>32; }
    }
    for(int i=0;i<8;i++) a[i]=t[i];
}

// Multiply a*b (8x8) -> 16-limb product prod[16]
__device__ __forceinline__ void mul_raw(const u32 a[8], const u32 b[8], u32 prod[16]){
    u64 r[16]; for(int i=0;i<16;i++) r[i]=0;
    for(int i=0;i<8;i++){
        u64 carry=0;
        for(int j=0;j<8;j++){
            u64 cur = r[i+j] + (u64)a[i]*b[j] + carry;
            r[i+j] = (u32)cur;
            carry = cur>>32;
        }
        r[i+8]+=carry;
    }
    for(int i=0;i<16;i++) prod[i]=(u32)r[i];
}

// Reduce a 512-bit value (16 limbs) mod p, result in out[8].
// 2^256 ≡ c (mod p), c = 2^32+977. Fold high half via c, iterate.
__device__ __forceinline__ void reduce512(u32 t[16], u32 out[8]){
    // Repeatedly: take limbs[8..] as H, low as L; result = L + H*c.
    // H*c = H*977 + H*2^32 (i.e. H shifted up 1 limb). We accumulate into a 9..10-limb buffer.
    // Do it generically: while there are nonzero limbs above index 7, fold.
    u32 w[16]; for(int i=0;i<16;i++) w[i]=t[i];
    for(int iter=0; iter<4; iter++){
        // H = w[8..15], L = w[0..7]; compute L + H*977 + (H<<32)
        // First check if H is zero
        bool hz=true; for(int i=8;i<16;i++) if(w[i]){hz=false;break;}
        if(hz) break;
        u64 acc[10]; for(int i=0;i<10;i++) acc[i]=0;
        for(int i=0;i<8;i++) acc[i]=w[i];        // L
        // + H*977  (H = w[8..15])
        u64 carry=0;
        for(int i=0;i<8;i++){
            u64 cur = acc[i] + (u64)w[8+i]*C_LOW + carry;
            acc[i]=(u32)cur; carry=cur>>32;
        }
        acc[8]+=carry;
        // + H<<32 (i.e. H added starting at limb 1)
        carry=0;
        for(int i=0;i<8;i++){
            u64 cur = acc[i+1] + (u64)w[8+i] + carry;
            acc[i+1]=(u32)cur; carry=cur>>32;
        }
        acc[9]+=carry;
        for(int i=0;i<16;i++) w[i] = (i<10)?(u32)acc[i]:0;
    }
    for(int i=0;i<8;i++) out[i]=w[i];
    if(geP(out)) subP(out);
    if(geP(out)) subP(out);
}

__device__ __forceinline__ void mulmod(const u32 a[8], const u32 b[8], u32 out[8]){
    u32 prod[16]; mul_raw(a,b,prod); reduce512(prod,out);
}
__device__ __forceinline__ void sqrmod(const u32 a[8], u32 out[8]){ mulmod(a,a,out); }
__device__ __forceinline__ void cpy(u32 d[8], const u32 s[8]){ for(int i=0;i<8;i++) d[i]=s[i]; }
__device__ __forceinline__ bool isZero(const u32 a[8]){ for(int i=0;i<8;i++) if(a[i]) return false; return true; }
__device__ __forceinline__ bool eq(const u32 a[8], const u32 b[8]){ for(int i=0;i<8;i++) if(a[i]!=b[i]) return false; return true; }

// Fermat inverse: a^(p-2) mod p. p-2 limbs:
__device__ __constant__ u32 PM2[8]={0xFFFFFC2D,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF};
__device__ void invmod(const u32 a[8], u32 out[8]){
    u32 r[8]={1,0,0,0,0,0,0,0}; u32 base[8]; cpy(base,a);
    for(int i=0;i<256;i++){
        if((PM2[i>>5]>>(i&31))&1){ u32 t[8]; mulmod(r,base,t); cpy(r,t); }
        u32 t[8]; sqrmod(base,t); cpy(base,t);
    }
    cpy(out,r);
}

// ---- EC Jacobian (a=0) ----
struct Jac { u32 X[8],Y[8],Z[8]; };
__device__ __forceinline__ bool jacInf(const Jac&p){ return isZero(p.Z); }

__device__ void jacDouble(const Jac&p, Jac&r){
    if(jacInf(p) || isZero(p.Y)){ for(int i=0;i<8;i++){r.X[i]=0;r.Y[i]=0;r.Z[i]=0;} return; }
    u32 YY[8],S[8],M[8],t[8],t2[8];
    sqrmod(p.Y,YY);
    mulmod(p.X,YY,S); u32 four[8]={4,0,0,0,0,0,0,0}; mulmod(S,four,S); // S=4*X*YY
    sqrmod(p.X,M); u32 three[8]={3,0,0,0,0,0,0,0}; mulmod(M,three,M); // M=3*X^2
    sqrmod(M,r.X); submod(r.X,S); submod(r.X,S);                      // X'=M^2-2S
    sqrmod(YY,t2); u32 eight[8]={8,0,0,0,0,0,0,0}; mulmod(t2,eight,t2);// 8*Y^4
    submod(S,r.X); mulmod(M,S,t); submod(t,t2); cpy(r.Y,t);           // Y'=M(S-X')-8Y^4
    u32 yz[8]; cpy(yz,p.Y); addmod(yz,p.Y); mulmod(yz,p.Z,r.Z);       // Z'=2*Y*Z
}

// mixed add: Jacobian p + affine (qx,qy)
__device__ void jacAddAff(const Jac&p, const u32 qx[8], const u32 qy[8], Jac&r){
    if(jacInf(p)){ cpy(r.X,qx); cpy(r.Y,qy); for(int i=0;i<8;i++) r.Z[i]=(i==0); return; }
    u32 Z1Z1[8],U2[8],S2[8],H[8],R[8],H2[8],H3[8],U1H2[8],t[8];
    sqrmod(p.Z,Z1Z1);
    mulmod(qx,Z1Z1,U2);
    mulmod(Z1Z1,p.Z,t); mulmod(qy,t,S2);
    // U1=p.X, S1=p.Y
    if(eq(p.X,U2)){
        if(eq(p.Y,S2)){ jacDouble(p,r); return; }
        for(int i=0;i<8;i++){r.X[i]=0;r.Y[i]=0;r.Z[i]=0;} return;
    }
    cpy(H,U2); submod(H,p.X);
    cpy(R,S2); submod(R,p.Y);
    sqrmod(H,H2); mulmod(H2,H,H3); mulmod(p.X,H2,U1H2);
    sqrmod(R,r.X); submod(r.X,H3); submod(r.X,U1H2); submod(r.X,U1H2);
    cpy(t,U1H2); submod(t,r.X); mulmod(R,t,r.Y); u32 s1h3[8]; mulmod(p.Y,H3,s1h3); submod(r.Y,s1h3);
    mulmod(p.Z,H,r.Z);
}

__device__ void jacToAff(const Jac&p, u32 ax[8], u32 ay[8]){
    if(jacInf(p)){ for(int i=0;i<8;i++){ax[i]=0;ay[i]=0;} return; }
    u32 zi[8],zi2[8],zi3[8]; invmod(p.Z,zi); sqrmod(zi,zi2); mulmod(zi2,zi,zi3);
    mulmod(p.X,zi2,ax); mulmod(p.Y,zi3,ay);
}

// double-and-add k*G for validation
__device__ void scalarMul(const u32 k[8], const u32 gx[8], const u32 gy[8], u32 rx[8], u32 ry[8]){
    Jac acc; for(int i=0;i<8;i++){acc.X[i]=0;acc.Y[i]=0;acc.Z[i]=0;}
    for(int i=255;i>=0;i--){
        Jac d; jacDouble(acc,d); acc=d;
        if((k[i>>5]>>(i&31))&1){ Jac a2; jacAddAff(acc,gx,gy,a2); acc=a2; }
    }
    jacToAff(acc,rx,ry);
}

// ============ Keccak-f1600 / SHAKE256 ============
#define ROTL64(x,n) (((x)<<(n))|((x)>>(64-(n))))
__device__ __constant__ int kf_rotc[24] = {1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44};
__device__ __constant__ int kf_piln[24] = {10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1};
__device__ __constant__ u64 kf_rndc[24] = {
 0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808aULL,0x8000000080008000ULL,
 0x000000000000808bULL,0x0000000080000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,
 0x000000000000008aULL,0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000aULL,
 0x000000008000808bULL,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,
 0x8000000000008002ULL,0x8000000000000080ULL,0x000000000000800aULL,0x800000008000000aULL,
 0x8000000080008081ULL,0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL};

__device__ void keccakf(u64 st[25]){
    u64 t, bc[5];
    for(int round=0; round<24; round++){
        for(int i=0;i<5;i++) bc[i]=st[i]^st[i+5]^st[i+10]^st[i+15]^st[i+20];
        for(int i=0;i<5;i++){ t = bc[(i+4)%5] ^ ROTL64(bc[(i+1)%5],1); for(int j=0;j<25;j+=5) st[j+i]^=t; }
        t=st[1];
        for(int i=0;i<24;i++){ int j=kf_piln[i]; bc[0]=st[j]; st[j]=ROTL64(t, kf_rotc[i]); t=bc[0]; }
        for(int j=0;j<25;j+=5){ for(int i=0;i<5;i++) bc[i]=st[j+i]; for(int i=0;i<5;i++) st[j+i]^=(~bc[(i+1)%5]) & bc[(i+2)%5]; }
        st[0]^=kf_rndc[round];
    }
}
// SHAKE256 (rate 136). Simple one-shot for validation.
__device__ void shake256(const uint8_t* in, int inlen, uint8_t* out, int outlen){
    u64 st[25]; for(int i=0;i<25;i++) st[i]=0;
    uint8_t* sb=(uint8_t*)st; const int rate=136; int pt=0;
    for(int i=0;i<inlen;i++){ sb[pt]^=in[i]; if(++pt==rate){ keccakf(st); pt=0; } }
    sb[pt]^=0x1F; sb[rate-1]^=0x80; keccakf(st);
    pt=0; for(int i=0;i<outlen;i++){ if(pt==rate){ keccakf(st); pt=0; } out[i]=sb[pt++]; }
}

// ================= Dialog-GCD filter (bit-exact port) =================
// Plain (non-mod) u256 bit helpers operating on u32 v[8], little-endian.
__device__ __forceinline__ int u_cmp(const u32 a[8], const u32 b[8]){ // 1 if a>b, -1 if a<b, 0 eq
    for(int i=7;i>=0;i--){ if(a[i]!=b[i]) return a[i]>b[i]?1:-1; }
    return 0;
}
__device__ __forceinline__ bool u_bit(const u32 a[8], int i){ return (a[i>>5]>>(i&31))&1u; }
__device__ __forceinline__ int u_bitlen(const u32 a[8]){
    for(int i=7;i>=0;i--){ if(a[i]) return i*32 + (32 - __clz(a[i])); }
    return 0;
}
// r = a & ((1<<width)-1)
__device__ __forceinline__ void u_low(const u32 a[8], u32 r[8], int width){
    for(int i=0;i<8;i++){ int lo=i*32, hi=lo+32;
        if(width>=hi) r[i]=a[i];
        else if(width<=lo) r[i]=0;
        else r[i]=a[i] & ((1u<<(width-lo))-1u);
    }
}
// r = a with low `width` bits cleared (high part)
__device__ __forceinline__ void u_high(const u32 a[8], u32 r[8], int width){
    for(int i=0;i<8;i++){ int lo=i*32, hi=lo+32;
        if(width>=hi) r[i]=0;
        else if(width<=lo) r[i]=a[i];
        else r[i]=a[i] & ~((1u<<(width-lo))-1u);
    }
}
// r = a >> n  (0<=n<256, logical)
__device__ __forceinline__ void u_shr(const u32 a[8], u32 r[8], int n){
    if(n>=256){ for(int i=0;i<8;i++) r[i]=0; return; }
    int wsh=n>>5, bsh=n&31;
    for(int i=0;i<8;i++){
        u32 lo=(i+wsh<8)?a[i+wsh]:0;
        u32 hi=(i+wsh+1<8)?a[i+wsh+1]:0;
        r[i]= bsh? ((lo>>bsh)|(hi<<(32-bsh))) : lo;
    }
}
__device__ __forceinline__ void u_shr1(u32 a[8]){
    for(int i=0;i<8;i++){ u32 hi=(i+1<8)?a[i+1]:0; a[i]=(a[i]>>1)|(hi<<31); }
}
__device__ __forceinline__ void u_sub(const u32 a[8], const u32 b[8], u32 r[8]){ // wrapping a-b
    u64 br=0; for(int i=0;i<8;i++){ u64 d=(u64)a[i]-b[i]-br; r[i]=(u32)d; br=(d>>63)&1; }
}
__device__ __forceinline__ void u_swap(u32 a[8], u32 b[8]){ for(int i=0;i<8;i++){ u32 t=a[i];a[i]=b[i];b[i]=t; } }

// shift_right_active(v, aw): keep low aw-bit window, shift that window right by 1.
__device__ __forceinline__ void shift_right_active(u32 v[8], int aw){
    u32 x[8],hi[8]; u_low(v,x,aw); u_shr1(x); u_high(v,hi,aw);
    for(int i=0;i<8;i++) v[i]=x[i]|hi[i];
}
// swap_active_except_bit0(u,v,aw)
__device__ __forceinline__ void swap_active_except_bit0(u32 u[8], u32 v[8], int aw){
    u32 ulo[8],vlo[8];
    u_low(u,ulo,aw); u_low(v,vlo,aw);            // low aw bits
    ulo[0]&=~1u; vlo[0]&=~1u;                     // clear bit0 -> mask_hi part
    u32 ub0=u[0]&1u, vb0=v[0]&1u;
    u32 uhi[8],vhi[8]; u_high(u,uhi,aw); u_high(v,vhi,aw); // bits >= aw stay put
    for(int i=0;i<8;i++){ u[i]=vlo[i]|uhi[i]; v[i]=ulo[i]|vhi[i]; }
    u[0]=(u[0]&~1u)|ub0; v[0]=(v[0]&~1u)|vb0;
}
__device__ __forceinline__ bool cmp_gt_window(const u32 u[8], const u32 v[8], int width){
    u32 a[8],b[8]; u_low(u,a,width); u_low(v,b,width); return u_cmp(a,b)>0;
}
__device__ __forceinline__ bool cmp_gt_truncated(const u32 u[8], const u32 v[8], int width, int cb){
    if(cb>width) cb=width; if(cb<1) cb=1; int lo=width-cb;
    u32 us[8],vs[8],a[8],b[8]; u_shr(u,us,lo); u_shr(v,vs,lo);
    u_low(us,a,cb); u_low(vs,b,cb); return u_cmp(a,b)>0;
}
// sub_low_window(v,u,width): v' = (v & ~mask) | ((v-u)&mask)
__device__ __forceinline__ void sub_low_window(u32 v[8], const u32 uu[8], int width){
    u32 vm[8],um[8],diff[8],hi[8]; u_low(v,vm,width); u_low(uu,um,width);
    u_sub(vm,um,diff); u_low(diff,diff,width); u_high(v,hi,width);
    for(int i=0;i<8;i++) v[i]=hi[i]|diff[i];
}

// cfg in constant memory
__device__ __constant__ int d_odd_u, d_k2, d_k2f0, d_active_iters, d_compare_bits;
__device__ __constant__ int d_aw[402], d_cb[402], d_bw[402];

__device__ void full_gcd_step(u32 u[8], u32 v[8]){
    bool b0=u_bit(v,0); bool fgt=u_cmp(u,v)>0;
    if(b0&&fgt){ if(d_odd_u) swap_active_except_bit0(u,v,256); else u_swap(u,v); }
    if(b0){ u_sub(v,u,v); if(d_odd_u) v[0]^=1u; }
    u_shr1(v);
    if(d_k2 && !d_k2f0){ if(!u_bit(v,0)) u_shr1(v); }
}
__device__ int full_gcd_steps_until_zero(const u32 uin[8], const u32 vin[8], int limit){
    u32 u[8],v[8]; cpy(u,uin); cpy(v,vin); int steps=0;
    while(!isZero(v) && steps<limit){ full_gcd_step(u,v); steps++; }
    return steps;
}
// returns true if HARD (overflow); advances u,v one truncated step
__device__ bool truncated_gcd_step(u32 u[8], u32 v[8], int step){
    int aw=d_aw[step];
    if(u_bitlen(u)>aw || u_bitlen(v)>aw) return true; // WidthOverflow
    int cb=d_cb[step];
    bool trunc_gt=cmp_gt_truncated(u,v,aw,cb);
    bool b0=u_bit(v,0);
    bool b0b1=b0 && trunc_gt;
    if(b0b1){ if(d_odd_u) swap_active_except_bit0(u,v,aw); else u_swap(u,v); }
    if(b0){ int body_w=d_bw[step];
        if(d_odd_u){ if(body_w<=1){ v[0]^=1u; } else { sub_low_window(v,u,body_w); v[0]^=1u; } }
        else { sub_low_window(v,u,body_w); }
    }
    shift_right_active(v,aw);
    if(d_k2 && !d_k2f0){ if(!u_bit(v,0)) shift_right_active(v,aw); }
    return false;
}
// returns true if factor is CLEAN (Ok), false if hard
__device__ __constant__ u32 d_P[8];
__device__ bool check_gcd_factor(const u32 factor[8]){
    if(isZero(factor)) return false;
    int steps=full_gcd_steps_until_zero(d_P, factor, d_active_iters+1);
    if(steps>d_active_iters) return false;
    u32 u[8],v[8]; cpy(u,d_P); cpy(v,factor);
    for(int step=0; step<d_active_iters; step++){
        if(truncated_gcd_step(u,v,step)) return false;
    }
    return true;
}

// ================= comb + per-nonce derivation =================
// comb table in global memory: 32*256 affine points, each 16 u32 (x[8],y[8]).
__device__ u32* d_comb;  // [32*256*16]
__device__ __forceinline__ const u32* comb_x(int j,int d){ return &d_comb[((j*256+d)*16)]; }
__device__ __forceinline__ const u32* comb_y(int j,int d){ return &d_comb[((j*256+d)*16)+8]; }

__device__ void comb_mul_jac(const u32 k[8], Jac& acc){
    for(int i=0;i<8;i++){ acc.X[i]=0; acc.Y[i]=0; acc.Z[i]=0; }
    for(int j=0;j<32;j++){
        u32 byte=(k[j>>2]>>((j&3)*8))&0xffu;
        if(byte){ Jac r; jacAddAff(acc, comb_x(j,byte), comb_y(j,byte), r); acc=r; }
    }
}
// submod a-b mod p (a,b<p) -> out  (== sub_mod_p)
__device__ __forceinline__ void submod_p(const u32 a[8], const u32 b[8], u32 out[8]){
    cpy(out,a); submod(out,b);
}

// base keccak state + params
__device__ __constant__ u64 d_base_st[25];
__device__ __constant__ int d_base_pt;
__device__ __constant__ u64 d_tx0, d_tx1;

__device__ void feed_x_op_dev(u64 st[25], int& pt, u64 q){
    uint8_t* sb=(uint8_t*)st; const int rate=136;
    const uint8_t kind=6; sb[pt]^=kind; if(++pt==rate){keccakf(st);pt=0;}
    u64 NO=~0ull;
    // q_control2, q_control1 = NO
    for(int b=0;b<8;b++){ sb[pt]^=(uint8_t)(NO>>(b*8)); if(++pt==rate){keccakf(st);pt=0;} }
    for(int b=0;b<8;b++){ sb[pt]^=(uint8_t)(NO>>(b*8)); if(++pt==rate){keccakf(st);pt=0;} }
    // q_target = q
    for(int b=0;b<8;b++){ sb[pt]^=(uint8_t)(q>>(b*8)); if(++pt==rate){keccakf(st);pt=0;} }
    // c_target, c_condition, r_target = NO
    for(int r=0;r<3;r++) for(int b=0;b<8;b++){ sb[pt]^=(uint8_t)(NO>>(b*8)); if(++pt==rate){keccakf(st);pt=0;} }
}

// squeeze state: reads bytes on demand
struct Squeezer { u64 st[25]; int pt; };
__device__ void squeeze_init(Squeezer& s, const u64 base[25], int base_pt, u64 nonce, u64 tx0, u64 tx1){
    for(int i=0;i<25;i++) s.st[i]=base[i];
    int pt=base_pt;
    for(int i=0;i<48;i++){ u64 q=((nonce>>i)&1)?tx1:tx0; feed_x_op_dev(s.st,pt,q); feed_x_op_dev(s.st,pt,q); }
    uint8_t* sb=(uint8_t*)s.st; sb[pt]^=0x1F; sb[135]^=0x80; keccakf(s.st);
    s.pt=0;
}
__device__ __forceinline__ void squeeze_bytes(Squeezer& s, uint8_t* out, int n){
    uint8_t* sb=(uint8_t*)s.st;
    for(int i=0;i<n;i++){ if(s.pt==136){ keccakf(s.st); s.pt=0; } out[i]=sb[s.pt++]; }
}

// returns true if nonce is CLEAN
__device__ bool nonce_is_clean(u64 nonce){
    Squeezer sq; squeeze_init(sq, d_base_st, d_base_pt, nonce, d_tx0, d_tx1);
    for(int shot=0; shot<9024; shot++){
        uint8_t rb[64]; squeeze_bytes(sq, rb, 64);
        u32 k1[8],k2[8];
        for(int i=0;i<8;i++){ k1[i]=rb[4*i]|(rb[4*i+1]<<8)|(rb[4*i+2]<<16)|(rb[4*i+3]<<24); }
        for(int i=0;i<8;i++){ k2[i]=rb[32+4*i]|(rb[32+4*i+1]<<8)|(rb[32+4*i+2]<<16)|(rb[32+4*i+3]<<24); }
        Jac tj,oj; comb_mul_jac(k1,tj); comb_mul_jac(k2,oj);
        u32 tx[8],ty[8],ox[8],oy[8];
        jacToAff(tj,tx,ty); jacToAff(oj,ox,oy);
        if(eq(tx,ox)) continue;
        if(isZero(tx)&&isZero(ty)) continue;
        if(isZero(ox)&&isZero(oy)) continue;
        u32 den[8]; submod_p(ox,tx,den);
        u32 deni[8]; invmod(den,deni);
        u32 num[8]; submod_p(oy,ty,num);
        u32 lambda[8]; mulmod(num,deni,lambda);
        u32 ex[8]; sqrmod(lambda,ex); submod(ex,tx); submod(ex,ox);
        u32 dx[8],c[8]; submod_p(tx,ox,dx); submod_p(ox,ex,c);
        if(!check_gcd_factor(dx)) return false;
        if(!check_gcd_factor(c)) return false;
    }
    return true;
}

__global__ void search_kernel(u64 start, u64 count, u32* out_cnt, u64* out_list, int max_out){
    u64 idx = blockIdx.x*(u64)blockDim.x + threadIdx.x;
    for(u64 i=idx; i<count; i += gridDim.x*(u64)blockDim.x){
        u64 nonce = start + i;
        if(nonce_is_clean(nonce)){
            u32 pos=atomicAdd(out_cnt,1u);
            if(pos<(u32)max_out) out_list[pos]=nonce;
        }
    }
}

// ===== shot-parallel: one BLOCK per nonce, WAVE threads split the 9024 shots =====
__device__ bool shot_is_hard(const u32 k1[8], const u32 k2[8]){
    Jac tj,oj; comb_mul_jac(k1,tj); comb_mul_jac(k2,oj);
    u32 tx[8],ty[8],ox[8],oy[8];
    jacToAff(tj,tx,ty); jacToAff(oj,ox,oy);
    if(eq(tx,ox)) return false;
    if(isZero(tx)&&isZero(ty)) return false;
    if(isZero(ox)&&isZero(oy)) return false;
    u32 den[8]; submod_p(ox,tx,den);
    u32 deni[8]; invmod(den,deni);
    u32 num[8]; submod_p(oy,ty,num);
    u32 lambda[8]; mulmod(num,deni,lambda);
    u32 ex[8]; sqrmod(lambda,ex); submod(ex,tx); submod(ex,ox);
    u32 dx[8],c[8]; submod_p(tx,ox,dx); submod_p(ox,ex,c);
    if(!check_gcd_factor(dx)) return true;
    if(!check_gcd_factor(c)) return true;
    return false;
}

#define WAVE 128
__global__ void search_kernel2(u64 start, u64 count, u32* out_cnt, u64* out_list, int max_out){
    __shared__ u64 sq_st[25];
    __shared__ int sq_pt;
    __shared__ unsigned char wbytes[WAVE*64];
    __shared__ int hard_flag;
    int t = threadIdx.x;
    for(u64 nidx = blockIdx.x; nidx < count; nidx += gridDim.x){
        u64 nonce = start + nidx;
        if(t==0){
            for(int i=0;i<25;i++) sq_st[i]=d_base_st[i];
            int pt=d_base_pt;
            for(int i=0;i<48;i++){ u64 q=((nonce>>i)&1)?d_tx1:d_tx0; feed_x_op_dev(sq_st,pt,q); feed_x_op_dev(sq_st,pt,q); }
            unsigned char* sb=(unsigned char*)sq_st; sb[pt]^=0x1F; sb[135]^=0x80; keccakf(sq_st);
            sq_pt=0; hard_flag=0;
        }
        __syncthreads();
        for(int base_shot=0; base_shot<9024; base_shot+=WAVE){
            if(hard_flag) break;
            int n_this = (9024-base_shot < WAVE) ? (9024-base_shot) : WAVE;
            if(t==0){
                unsigned char* sb=(unsigned char*)sq_st;
                int need=n_this*64;
                for(int i=0;i<need;i++){ if(sq_pt==136){ keccakf(sq_st); sq_pt=0; } wbytes[i]=sb[sq_pt++]; }
            }
            __syncthreads();
            if(t < n_this){
                unsigned char* rb=&wbytes[t*64];
                u32 k1[8],k2[8];
                for(int i=0;i<8;i++) k1[i]=rb[4*i]|(rb[4*i+1]<<8)|(rb[4*i+2]<<16)|(rb[4*i+3]<<24);
                for(int i=0;i<8;i++) k2[i]=rb[32+4*i]|(rb[32+4*i+1]<<8)|(rb[32+4*i+2]<<16)|(rb[32+4*i+3]<<24);
                if(shot_is_hard(k1,k2)) hard_flag=1;
            }
            __syncthreads();
        }
        if(t==0 && !hard_flag){
            u32 pos=atomicAdd(out_cnt,1u);
            if(pos<(u32)max_out) out_list[pos]=nonce;
        }
        __syncthreads();
    }
}

// validate kernel: derive first (k1,k2) for probe nonce, write to out (16 u32)
__global__ void probe_kernel(u64 nonce, u32* out){
    Squeezer sq; squeeze_init(sq, d_base_st, d_base_pt, nonce, d_tx0, d_tx1);
    uint8_t rb[64]; squeeze_bytes(sq, rb, 64);
    for(int i=0;i<8;i++) out[i]=rb[4*i]|(rb[4*i+1]<<8)|(rb[4*i+2]<<16)|(rb[4*i+3]<<24);
    for(int i=0;i<8;i++) out[8+i]=rb[32+4*i]|(rb[32+4*i+1]<<8)|(rb[32+4*i+2]<<16)|(rb[32+4*i+3]<<24);
}

// ================= host =================
#include <cstdlib>
#include <cstring>
static u64 rd64(FILE*f){ u64 v; fread(&v,8,1,f); return v; }
static u32 rd32(FILE*f){ u32 v; fread(&v,4,1,f); return v; }

int main(int argc, char** argv){
    const char* dump = getenv("GPU_STATE"); if(!dump) dump="/tmp/gpu_state.bin";
    u64 start = argc>1? strtoull(argv[1],0,10):0;
    u64 count = argc>2? strtoull(argv[2],0,10):1;
    bool do_probe = (argc>3 && strcmp(argv[3],"probe")==0);

    FILE* f=fopen(dump,"rb"); if(!f){ printf("cannot open %s\n",dump); return 1; }
    u32 magic=rd32(f); if(magic!=0x47505531){ printf("bad magic %08x\n",magic); return 1; }
    u64 n_ops=rd64(f); u64 tx0=rd64(f), tx1=rd64(f);
    u32 base_pt=rd32(f);
    u64 base_st[25]; for(int i=0;i<25;i++) base_st[i]=rd64(f);
    u32 odd_u=rd32(f), k2=rd32(f), k2f0=rd32(f), ai=rd32(f), cbits=rd32(f);
    int aw[402],cb[402],bw[402]; for(int i=0;i<402;i++){aw[i]=0;cb[i]=0;bw[i]=0;}
    for(u32 i=0;i<ai;i++) aw[i]=(int)rd32(f);
    for(u32 i=0;i<ai;i++) cb[i]=(int)rd32(f);
    for(u32 i=0;i<ai;i++) bw[i]=(int)rd32(f);
    static u32 comb[32*256*16];
    fread(comb, 4, 32*256*16, f);
    u64 probe=rd64(f); u32 pk1[8],pk2[8]; fread(pk1,4,8,f); fread(pk2,4,8,f);
    fclose(f);
    printf("loaded: n_ops=%llu tx0=%llu tx1=%llu base_pt=%u ai=%u cbits=%u odd_u=%u k2=%u\n",
        (unsigned long long)n_ops,(unsigned long long)tx0,(unsigned long long)tx1,base_pt,ai,cbits,odd_u,k2);

    // upload constants
    u32 Phost[8]={0xFFFFFC2F,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF};
    cudaMemcpyToSymbol(d_P,Phost,32);
    int iodd=odd_u,ik2=k2,ik2f0=k2f0,iai=ai,icb=cbits;
    cudaMemcpyToSymbol(d_odd_u,&iodd,4); cudaMemcpyToSymbol(d_k2,&ik2,4);
    cudaMemcpyToSymbol(d_k2f0,&ik2f0,4); cudaMemcpyToSymbol(d_active_iters,&iai,4);
    cudaMemcpyToSymbol(d_compare_bits,&icb,4);
    cudaMemcpyToSymbol(d_aw,aw,sizeof(aw)); cudaMemcpyToSymbol(d_cb,cb,sizeof(cb)); cudaMemcpyToSymbol(d_bw,bw,sizeof(bw));
    cudaMemcpyToSymbol(d_base_st,base_st,200); cudaMemcpyToSymbol(d_base_pt,&base_pt,4);
    u64 utx0=tx0,utx1=tx1; cudaMemcpyToSymbol(d_tx0,&utx0,8); cudaMemcpyToSymbol(d_tx1,&utx1,8);
    u32* dcomb; cudaMalloc(&dcomb, sizeof(comb)); cudaMemcpy(dcomb,comb,sizeof(comb),cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_comb,&dcomb,sizeof(dcomb));

    if(do_probe){
        u32* dout; cudaMalloc(&dout,16*4);
        probe_kernel<<<1,1>>>(probe,dout);
        cudaError_t e=cudaDeviceSynchronize(); if(e){printf("probe err %s\n",cudaGetErrorString(e));return 1;}
        u32 h[16]; cudaMemcpy(h,dout,16*4,cudaMemcpyDeviceToHost);
        bool ok1=true,ok2=true; for(int i=0;i<8;i++){ if(h[i]!=pk1[i])ok1=false; if(h[8+i]!=pk2[i])ok2=false; }
        printf("probe nonce=%llu k1:%s k2:%s\n",(unsigned long long)probe, ok1?"OK":"MISMATCH", ok2?"OK":"MISMATCH");
        printf("k1 got "); for(int i=7;i>=0;i--) printf("%08x",h[i]); printf("\n   exp "); for(int i=7;i>=0;i--) printf("%08x",pk1[i]); printf("\n");
        return 0;
    }

    u32* dcnt; cudaMalloc(&dcnt,4); cudaMemset(dcnt,0,4);
    const int MAXOUT=4096; u64* dlist; cudaMalloc(&dlist,MAXOUT*8);
    bool k2mode = getenv("KERNEL2")!=NULL;
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1); cudaEventRecord(t0);
    if(k2mode){
        int blocks = getenv("BLOCKS")? atoi(getenv("BLOCKS")) : 512;
        search_kernel2<<<blocks,WAVE>>>(start,count,dcnt,dlist,MAXOUT);
    } else {
        search_kernel<<<256,128>>>(start,count,dcnt,dlist,MAXOUT);
    }
    cudaError_t e=cudaDeviceSynchronize(); if(e){printf("search err %s\n",cudaGetErrorString(e));return 1;}
    cudaEventRecord(t1); cudaEventSynchronize(t1); float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    u32 cnt; cudaMemcpy(&cnt,dcnt,4,cudaMemcpyDeviceToHost);
    u64 list[MAXOUT]; cudaMemcpy(list,dlist,(cnt<MAXOUT?cnt:MAXOUT)*8,cudaMemcpyDeviceToHost);
    for(u32 i=0;i<cnt && i<MAXOUT;i++) printf("CLEAN nonce=%llu\n",(unsigned long long)list[i]);
    printf("scanned %llu in %.2fs (%.0f nonce/s); clean=%u\n",
        (unsigned long long)count, ms/1000.0, count/(ms/1000.0), cnt);
    return 0;
}
