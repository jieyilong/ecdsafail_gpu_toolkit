// CUDA secp256k1 island searcher — gpu_island3.cu
// Batch-inversion shot-parallel kernel.  KERNEL3=1 activates.
//
// Based on gpu_island2.cu with Montgomery batch inversion for Z→affine
// and den inversions.  Three batch inversions per wave replace 384
// individual Fermat invmods with 3 Fermat invmods + ~768 mulmods.
//
// Shared memory per block: ~42 KB (fits 48 KB default on sm_89 RTX 4070).
//   Keccak state:      200 + 4 + 8192 + 4  = 8,400 B
//   s_z  [W][8]:       4,096 B  (reused for tj.Z, oj.Z, den)
//   s_a  [W][8]:       4,096 B  (tx, then ox)
//   s_b  [W][8]:       4,096 B  (ty, then oy)
//   s_jxy[W][16]:      8,192 B  (Jacobian X[8]||Y[8] for tj then oj)
//   s_pf [W][8]:       4,096 B  (batch-inv scratch: prefix products)
//   s_or [W][8]:       4,096 B  (batch-inv scratch: originals backup)
//   s_skip[W]:           512 B
//   Total:             ~41.9 KB

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

typedef uint32_t u32;
typedef uint64_t u64;

__device__ __constant__ u32 P[8] = {0xFFFFFC2F,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF};
#define C_LOW 0x000003D1u

__device__ __forceinline__ bool geP(const u32 a[8]){
    for(int i=7;i>=0;i--){ if(a[i]!=P[i]) return a[i]>P[i]; } return true;
}
__device__ __forceinline__ void subP(u32 a[8]){
    u64 b=0; for(int i=0;i<8;i++){ u64 t=(u64)a[i]-P[i]-b; a[i]=(u32)t; b=(t>>63)&1; }
}
__device__ __forceinline__ void addmod(u32 a[8], const u32 b[8]){
    u64 c=0; for(int i=0;i<8;i++){ u64 t=(u64)a[i]+b[i]+c; a[i]=(u32)t; c=t>>32; }
    if(c || geP(a)) subP(a);
}
__device__ __forceinline__ void submod(u32 a[8], const u32 b[8]){
    u64 br=0; u32 t[8];
    for(int i=0;i<8;i++){ u64 d=(u64)a[i]-b[i]-br; t[i]=(u32)d; br=(d>>63)&1; }
    if(br){ u64 c=0; for(int i=0;i<8;i++){ u64 s=(u64)t[i]+P[i]+c; t[i]=(u32)s; c=s>>32; } }
    for(int i=0;i<8;i++) a[i]=t[i];
}
// === 8x8-limb schoolbook multiply producing 512-bit result ===
__device__ __forceinline__ void mul_raw(const u32 a[8], const u32 b[8], u32 r[16]){
    for(int i=0;i<16;i++) r[i]=0;
    for(int i=0;i<8;i++){
        u64 carry=0;
        for(int j=0;j<8;j++){
            u64 cur = (u64)r[i+j] + (u64)a[i]*b[j] + carry;
            r[i+j] = (u32)cur;
            carry = cur >> 32;
        }
        r[i+8] += (u32)carry;
    }
}

// === Streamlined secp256k1 Solinas reduction: p = 2^256 - c, c = 2^32 + 0x3D1 ===
// Reduces a 512-bit value mod p in exactly 2 passes, no loops, no branches.
__device__ __forceinline__ void reduce_secp256k1(const u32 prod[16], u32 out[8]){
    // Pass 1: result = low256 + high256 * c  (c = 2^32 + 0x3D1)
    // high256 * c = high256 * 0x3D1 + high256 << 32
    u32 t[9];
    u64 acc = 0;
    
    // Limb 0: L[0] + H[0]*0x3D1
    acc = (u64)prod[0] + (u64)prod[8] * 0x3D1u;
    t[0] = (u32)acc; acc >>= 32;
    
    // Limbs 1-7: L[i] + H[i]*0x3D1 + H[i-1]  (H[i-1] is the <<32 shift)
    for(int i=1;i<8;i++){
        acc += (u64)prod[i] + (u64)prod[8+i] * 0x3D1u + (u64)prod[8+i-1];
        t[i] = (u32)acc; acc >>= 32;
    }
    // Limb 8: H[7] (from shift) + carry
    acc += (u64)prod[15];
    t[8] = (u32)acc;
    // acc>>32 is guaranteed 0 here
    
    // Pass 2: fold t[8] (at most ~33 bits) * c back into t[0..7]
    acc = (u64)t[0] + (u64)t[8] * 0x3D1u;
    out[0] = (u32)acc; acc >>= 32;
    
    // Limb 1: t[1] + t[8] (the <<32 component of t[8]*c)
    acc += (u64)t[1] + (u64)t[8];
    out[1] = (u32)acc; acc >>= 32;
    
    // Limbs 2-7: just propagate carry
    for(int i=2;i<8;i++){
        acc += (u64)t[i];
        out[i] = (u32)acc; acc >>= 32;
    }
    
    // Final: at most 1-2 conditional subtractions
    if(acc || geP(out)) subP(out);
    if(geP(out)) subP(out);
}

__device__ __forceinline__ void mulmod(const u32 a[8], const u32 b[8], u32 out[8]){
    u32 prod[16]; mul_raw(a,b,prod); reduce_secp256k1(prod,out);
}

// === Dedicated squaring: exploits a[i]*a[j] == a[j]*a[i] symmetry ===
// Cross-products computed once and doubled, diagonal terms added separately.
// Saves ~40% multiply instructions vs generic mulmod(a,a).
__device__ __forceinline__ void sqr_raw(const u32 a[8], u32 r[16]){
    // First compute cross products (i<j terms only)
    for(int i=0;i<16;i++) r[i]=0;
    for(int i=0;i<8;i++){
        u64 carry=0;
        for(int j=i+1;j<8;j++){
            u64 cur = (u64)r[i+j] + (u64)a[i]*a[j] + carry;
            r[i+j] = (u32)cur;
            carry = cur >> 32;
        }
        r[i+8] += (u32)carry;
    }
    // Double the cross products (left shift entire 512-bit value by 1)
    u32 top = 0;
    for(int i=0;i<16;i++){
        u32 nxt = r[i] >> 31;
        r[i] = (r[i] << 1) | top;
        top = nxt;
    }
    // Add diagonal terms a[i]^2
    u64 carry = 0;
    for(int i=0;i<8;i++){
        u64 sq = (u64)a[i] * a[i];
        u64 s = (u64)r[2*i] + (u32)sq + carry;
        r[2*i] = (u32)s;
        s = (u64)r[2*i+1] + (sq >> 32) + (s >> 32);
        r[2*i+1] = (u32)s;
        carry = s >> 32;
    }
}
__device__ __forceinline__ void sqrmod(const u32 a[8], u32 out[8]){
    u32 prod[16]; sqr_raw(a,prod); reduce_secp256k1(prod,out);
}
__device__ __forceinline__ void cpy(u32 d[8], const u32 s[8]){ for(int i=0;i<8;i++) d[i]=s[i]; }
__device__ __forceinline__ bool isZero(const u32 a[8]){ for(int i=0;i<8;i++) if(a[i]) return false; return true; }
__device__ __forceinline__ bool eq(const u32 a[8], const u32 b[8]){ for(int i=0;i<8;i++) if(a[i]!=b[i]) return false; return true; }

// secp256k1-optimized addition chain for a^(p-2) mod p.
// p-2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
// Exploits long runs of 1-bits. Only ~15 multiplies + 256 squarings.
// Helper: repeated squaring
__device__ __forceinline__ void sqrN(const u32 a[8], u32 out[8], int n){
    u32 t[8]; cpy(t, a);
    for(int i=0;i<n;i++){ u32 s[8]; sqrmod(t,s); cpy(t,s); }
    cpy(out,t);
}
__device__ void invmod(const u32 a[8], u32 out[8]){
    u32 tmp[8];
    
    // x2 = a^(2^2 - 1) = a^3
    u32 x2[8];
    sqrmod(a,tmp);
    mulmod(tmp,a,x2);
    
    // x3 = a^(2^3 - 1) = a^7
    u32 x3[8];
    sqrmod(x2,tmp);
    mulmod(tmp,a,x3);
    
    // x6 = a^(2^6 - 1)
    u32 x6[8];
    sqrN(x3,tmp,3);
    mulmod(tmp,x3,x6);                               // x6 = a^(2^6-1)
    
    // x9 = a^(2^9 - 1)
    u32 x9[8];
    sqrN(x6,tmp,3);
    mulmod(tmp,x3,x9);                               // x9 = a^(2^9-1)
    
    // x11 = a^(2^11 - 1)
    u32 x11[8];
    sqrN(x9,tmp,2);
    mulmod(tmp,x2,x11);                              // x11 = a^(2^11-1)
    
    // x22 = a^(2^22 - 1)
    u32 x22[8];
    sqrN(x11,tmp,11);
    mulmod(tmp,x11,x22);                             // x22 = a^(2^22-1)
    
    // x44 = a^(2^44 - 1)
    u32 x44[8];
    sqrN(x22,tmp,22);
    mulmod(tmp,x22,x44);                             // x44 = a^(2^44-1)
    
    // x88 = a^(2^88 - 1)
    u32 x88[8];
    sqrN(x44,tmp,44);
    mulmod(tmp,x44,x88);                             // x88 = a^(2^88-1)
    
    // x176 = a^(2^176 - 1)
    u32 x176[8];
    sqrN(x88,tmp,88);
    mulmod(tmp,x88,x176);                            // x176 = a^(2^176-1)
    
    // x220 = a^(2^220 - 1)
    u32 x220[8];
    sqrN(x176,tmp,44);
    mulmod(tmp,x44,x220);                            // x220 = a^(2^220-1)
    
    // x223 = a^(2^223 - 1)
    u32 x223[8];
    sqrN(x220,tmp,3);
    mulmod(tmp,x3,x223);                             // x223 = a^(2^223-1)
    
    // Now build the final exponent:
    // t = x223 * 2^23 = a^(2^246 - 2^23)
    u32 r[8];
    sqrN(x223,r,23);
    // t = t * x22 = a^(2^246 - 2^23 + 2^22 - 1) = a^(2^246 - 2^22 - 1)... 
    // Actually the precise exponent construction:
    // p-2 = 2^256 - 2^32 - 979
    // = 2^256 - 2^32 - 2^10 + 2^6 - 2^4 + 2^3 + 2^2 + 2^0 + ...
    // Standard libsecp256k1 chain:
    // After x223, square 23 times and multiply by x22:
    mulmod(r,x22,tmp); cpy(r,tmp);                   // a^(2^246 - 1)
    
    // Square 5 times, multiply by a:
    sqrN(r,tmp,5);
    mulmod(tmp,a,r);                                  // a^(2^251 - 2^5 + 1)
    
    // Square 3 times, multiply by x2:
    sqrN(r,tmp,3);
    mulmod(tmp,x2,r);                                 // a^(2^254 - 2^8 + 2^3 + ... )
    
    // Square 2 times, multiply by a:
    sqrN(r,tmp,2);
    mulmod(tmp,a,out);                                // a^(p-2)
}

struct Jac { u32 X[8],Y[8],Z[8]; };
__device__ __forceinline__ bool jacInf(const Jac&p){ return isZero(p.Z); }
__device__ void jacDouble(const Jac&p, Jac&r){
    if(jacInf(p)||isZero(p.Y)){ for(int i=0;i<8;i++){r.X[i]=0;r.Y[i]=0;r.Z[i]=0;} return; }
    u32 YY[8],S[8],M[8],t[8],t2[8];
    sqrmod(p.Y,YY);
    mulmod(p.X,YY,S); addmod(S,S); addmod(S,S);  // S *= 4 via 2 doublings
    sqrmod(p.X,M); cpy(t,M); addmod(M,t); addmod(M,t);  // M = 3*X² via M+M+M
    sqrmod(M,r.X); submod(r.X,S); submod(r.X,S);
    sqrmod(YY,t2); addmod(t2,t2); addmod(t2,t2); addmod(t2,t2);  // t2 *= 8 via 3 doublings
    submod(S,r.X); mulmod(M,S,t); submod(t,t2); cpy(r.Y,t);
    u32 yz[8]; cpy(yz,p.Y); addmod(yz,p.Y); mulmod(yz,p.Z,r.Z);
}
__device__ void jacAddAff(const Jac&p, const u32 qx[8], const u32 qy[8], Jac&r){
    if(jacInf(p)){ cpy(r.X,qx); cpy(r.Y,qy); for(int i=0;i<8;i++) r.Z[i]=(i==0); return; }
    u32 Z1Z1[8],U2[8],S2[8],H[8],R[8],H2[8],H3[8],U1H2[8],t[8];
    sqrmod(p.Z,Z1Z1); mulmod(qx,Z1Z1,U2);
    mulmod(Z1Z1,p.Z,t); mulmod(qy,t,S2);
    if(eq(p.X,U2)){
        if(eq(p.Y,S2)){ jacDouble(p,r); return; }
        for(int i=0;i<8;i++){r.X[i]=0;r.Y[i]=0;r.Z[i]=0;} return;
    }
    cpy(H,U2); submod(H,p.X); cpy(R,S2); submod(R,p.Y);
    sqrmod(H,H2); mulmod(H2,H,H3); mulmod(p.X,H2,U1H2);
    sqrmod(R,r.X); submod(r.X,H3); submod(r.X,U1H2); submod(r.X,U1H2);
    cpy(t,U1H2); submod(t,r.X); mulmod(R,t,r.Y);
    u32 s1h3[8]; mulmod(p.Y,H3,s1h3); submod(r.Y,s1h3);
    mulmod(p.Z,H,r.Z);
}

#define ROTL64(x,n) (((x)<<(n))|((x)>>(64-(n))))
__device__ __constant__ int kf_rotc[24]={1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44};
__device__ __constant__ int kf_piln[24]={10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1};
__device__ __constant__ u64 kf_rndc[24]={
 0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808aULL,0x8000000080008000ULL,
 0x000000000000808bULL,0x0000000080000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,
 0x000000000000008aULL,0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000aULL,
 0x000000008000808bULL,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,
 0x8000000000008002ULL,0x8000000000000080ULL,0x000000000000800aULL,0x800000008000000aULL,
 0x8000000080008081ULL,0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL};
__device__ void keccakf(u64 st[25]){
    u64 t,bc[5];
    for(int round=0;round<24;round++){
        for(int i=0;i<5;i++) bc[i]=st[i]^st[i+5]^st[i+10]^st[i+15]^st[i+20];
        for(int i=0;i<5;i++){t=bc[(i+4)%5]^ROTL64(bc[(i+1)%5],1); for(int j=0;j<25;j+=5) st[j+i]^=t;}
        t=st[1]; for(int i=0;i<24;i++){int j=kf_piln[i]; bc[0]=st[j]; st[j]=ROTL64(t,kf_rotc[i]); t=bc[0];}
        for(int j=0;j<25;j+=5){for(int i=0;i<5;i++) bc[i]=st[j+i]; for(int i=0;i<5;i++) st[j+i]^=(~bc[(i+1)%5])&bc[(i+2)%5];}
        st[0]^=kf_rndc[round];
    }
}

__device__ __forceinline__ int u_cmp(const u32 a[8], const u32 b[8]){
    for(int i=7;i>=0;i--){if(a[i]!=b[i]) return a[i]>b[i]?1:-1;} return 0;
}
__device__ __forceinline__ bool u_bit(const u32 a[8], int i){ return (a[i>>5]>>(i&31))&1u; }
__device__ __forceinline__ int u_bitlen(const u32 a[8]){
    for(int i=7;i>=0;i--){if(a[i]) return i*32+(32-__clz(a[i]));} return 0;
}
__device__ __forceinline__ void u_low(const u32 a[8], u32 r[8], int w){
    for(int i=0;i<8;i++){int lo=i*32,hi=lo+32; if(w>=hi) r[i]=a[i]; else if(w<=lo) r[i]=0; else r[i]=a[i]&((1u<<(w-lo))-1u);}
}
__device__ __forceinline__ void u_high(const u32 a[8], u32 r[8], int w){
    for(int i=0;i<8;i++){int lo=i*32,hi=lo+32; if(w>=hi) r[i]=0; else if(w<=lo) r[i]=a[i]; else r[i]=a[i]&~((1u<<(w-lo))-1u);}
}
__device__ __forceinline__ void u_shr(const u32 a[8], u32 r[8], int n){
    if(n>=256){for(int i=0;i<8;i++) r[i]=0; return;}
    int wsh=n>>5,bsh=n&31;
    for(int i=0;i<8;i++){u32 lo=(i+wsh<8)?a[i+wsh]:0; u32 hi=(i+wsh+1<8)?a[i+wsh+1]:0; r[i]=bsh?((lo>>bsh)|(hi<<(32-bsh))):lo;}
}
__device__ __forceinline__ void u_shr1(u32 a[8]){ for(int i=0;i<8;i++){u32 hi=(i+1<8)?a[i+1]:0; a[i]=(a[i]>>1)|(hi<<31);} }
__device__ __forceinline__ void u_sub(const u32 a[8], const u32 b[8], u32 r[8]){
    u64 br=0; for(int i=0;i<8;i++){u64 d=(u64)a[i]-b[i]-br; r[i]=(u32)d; br=(d>>63)&1;}
}
__device__ __forceinline__ void u_swap(u32 a[8], u32 b[8]){ for(int i=0;i<8;i++){u32 t=a[i];a[i]=b[i];b[i]=t;} }
__device__ __forceinline__ void shift_right_active(u32 v[8], int aw){
    u32 x[8],hi[8]; u_low(v,x,aw); u_shr1(x); u_high(v,hi,aw); for(int i=0;i<8;i++) v[i]=x[i]|hi[i];
}
__device__ __forceinline__ void swap_active_except_bit0(u32 u[8], u32 v[8], int aw){
    u32 ulo[8],vlo[8]; u_low(u,ulo,aw); u_low(v,vlo,aw);
    ulo[0]&=~1u; vlo[0]&=~1u; u32 ub0=u[0]&1u,vb0=v[0]&1u;
    u32 uhi[8],vhi[8]; u_high(u,uhi,aw); u_high(v,vhi,aw);
    for(int i=0;i<8;i++){u[i]=vlo[i]|uhi[i]; v[i]=ulo[i]|vhi[i];}
    u[0]=(u[0]&~1u)|ub0; v[0]=(v[0]&~1u)|vb0;
}
__device__ __forceinline__ bool cmp_gt_truncated(const u32 u[8], const u32 v[8], int w, int cb){
    if(cb>w) cb=w; if(cb<1) cb=1; int lo=w-cb;
    u32 us[8],vs[8],a[8],b[8]; u_shr(u,us,lo); u_shr(v,vs,lo); u_low(us,a,cb); u_low(vs,b,cb); return u_cmp(a,b)>0;
}
__device__ __forceinline__ void sub_low_window(u32 v[8], const u32 uu[8], int w){
    u32 vm[8],um[8],diff[8],hi[8]; u_low(v,vm,w); u_low(uu,um,w);
    u_sub(vm,um,diff); u_low(diff,diff,w); u_high(v,hi,w); for(int i=0;i<8;i++) v[i]=hi[i]|diff[i];
}

__device__ __constant__ int d_odd_u, d_k2, d_k2f0, d_active_iters, d_compare_bits;
__device__ __constant__ int d_aw[402], d_cb[402], d_bw[402];
__device__ void full_gcd_step(u32 u[8], u32 v[8]){
    bool b0=u_bit(v,0); bool fgt=u_cmp(u,v)>0;
    if(b0&&fgt){ if(d_odd_u) swap_active_except_bit0(u,v,256); else u_swap(u,v); }
    if(b0){ u_sub(v,u,v); if(d_odd_u) v[0]^=1u; }
    u_shr1(v); if(d_k2&&!d_k2f0){if(!u_bit(v,0)) u_shr1(v);}
}
__device__ int full_gcd_steps_until_zero(const u32 uin[8], const u32 vin[8], int limit){
    u32 u[8],v[8]; cpy(u,uin); cpy(v,vin); int steps=0;
    while(!isZero(v)&&steps<limit){full_gcd_step(u,v); steps++;} return steps;
}
__device__ bool truncated_gcd_step(u32 u[8], u32 v[8], int step){
    int aw=d_aw[step];
    if(u_bitlen(u)>aw||u_bitlen(v)>aw) return true;
    int cb=d_cb[step]; bool trunc_gt=cmp_gt_truncated(u,v,aw,cb);
    bool b0=u_bit(v,0); bool b0b1=b0&&trunc_gt;
    if(b0b1){ if(d_odd_u) swap_active_except_bit0(u,v,aw); else u_swap(u,v); }
    if(b0){ int body_w=d_bw[step];
        if(d_odd_u){if(body_w<=1){v[0]^=1u;} else {sub_low_window(v,u,body_w); v[0]^=1u;}}
        else {sub_low_window(v,u,body_w);}
    }
    shift_right_active(v,aw); if(d_k2&&!d_k2f0){if(!u_bit(v,0)) shift_right_active(v,aw);}
    return false;
}
__device__ __constant__ u32 d_P[8];
__device__ bool check_gcd_factor(const u32 factor[8]){
    if(isZero(factor)) return false;
    int steps=full_gcd_steps_until_zero(d_P,factor,d_active_iters+1);
    if(steps>d_active_iters) return false;
    u32 u[8],v[8]; cpy(u,d_P); cpy(v,factor);
    for(int step=0;step<d_active_iters;step++){if(truncated_gcd_step(u,v,step)) return false;}
    return true;
}

// ===== Rejection instrumentation =====
// 10 atomic u64 counters in global memory, toggled by INSTRUMENT=1 env var.
// Overhead: ~1 atomicAdd per rejected nonce (hard_flag kills the block).
struct RejCounters {
    u64 dx_zero, dx_noconv, dx_width;
    u64 c_zero, c_noconv, c_width;
    u64 nonces_launched, nonces_clean;
    u64 first_dx, first_c;
};
__device__ __constant__ RejCounters* d_rej;

// Returns 0 if clean, rejection category 1-6 if hard.
// factor_id: 0=dx, 1=c.  category: +1=zero, +2=noconv, +3=width.
__device__ int check_gcd_factor_inst(const u32 factor[8], int factor_id){
    unsigned long long* base = (unsigned long long*)&d_rej->dx_zero + factor_id*3;
    if(isZero(factor)){
        atomicAdd(base+0, 1ULL);
        return factor_id*3+1;
    }
    int steps=full_gcd_steps_until_zero(d_P,factor,d_active_iters+1);
    if(steps>d_active_iters){
        atomicAdd(base+1, 1ULL);
        return factor_id*3+2;
    }
    u32 u[8],v[8]; cpy(u,d_P); cpy(v,factor);
    for(int step=0;step<d_active_iters;step++){
        if(truncated_gcd_step(u,v,step)){
            atomicAdd(base+2, 1ULL);
            return factor_id*3+3;
        }
    }
    return 0;
}

__device__ u32* d_comb;
__device__ __forceinline__ const u32* comb_x(int j,int d){ return &d_comb[((j*256+d)*16)]; }
__device__ __forceinline__ const u32* comb_y(int j,int d){ return &d_comb[((j*256+d)*16)+8]; }
__device__ void comb_mul_jac(const u32 k[8], Jac& acc){
    for(int i=0;i<8;i++){acc.X[i]=0;acc.Y[i]=0;acc.Z[i]=0;}
    for(int j=0;j<32;j++){u32 byte=(k[j>>2]>>((j&3)*8))&0xffu; if(byte){Jac r; jacAddAff(acc,comb_x(j,byte),comb_y(j,byte),r); acc=r;}}
}
__device__ __forceinline__ void submod_p(const u32 a[8], const u32 b[8], u32 out[8]){ cpy(out,a); submod(out,b); }

__device__ __constant__ u64 d_base_st[25];
__device__ __constant__ int d_base_pt;
__device__ __constant__ u64 d_tx0, d_tx1;
__device__ void feed_x_op_dev(u64 st[25], int& pt, u64 q){
    uint8_t* sb=(uint8_t*)st; const int rate=136;
    const uint8_t kind=6; sb[pt]^=kind; if(++pt==rate){keccakf(st);pt=0;}
    u64 NO=~0ull;
    for(int b=0;b<8;b++){sb[pt]^=(uint8_t)(NO>>(b*8)); if(++pt==rate){keccakf(st);pt=0;}}
    for(int b=0;b<8;b++){sb[pt]^=(uint8_t)(NO>>(b*8)); if(++pt==rate){keccakf(st);pt=0;}}
    for(int b=0;b<8;b++){sb[pt]^=(uint8_t)(q>>(b*8)); if(++pt==rate){keccakf(st);pt=0;}}
    for(int r=0;r<3;r++) for(int b=0;b<8;b++){sb[pt]^=(uint8_t)(NO>>(b*8)); if(++pt==rate){keccakf(st);pt=0;}}
}

// ===== Warp-level Montgomery batch inversion =====
// Inverts 32 values in val[8] across a warp in-place.
// No __syncthreads() needed. Uses parallel prefix and suffix products.
__device__ void warp_batch_inv(u32 val[8]){
    int lane = threadIdx.x & 31;
    
    // 1. Inclusive prefix product P
    u32 P[8]; cpy(P, val);
    u32 tmp[8], tmp2[8];
    for(int i=0;i<8;i++) tmp[i] = __shfl_up_sync(0xffffffff, P[i], 1);
    if(lane >= 1){ mulmod(P, tmp, tmp2); cpy(P, tmp2); }
    
    for(int i=0;i<8;i++) tmp[i] = __shfl_up_sync(0xffffffff, P[i], 2);
    if(lane >= 2){ mulmod(P, tmp, tmp2); cpy(P, tmp2); }
    
    for(int i=0;i<8;i++) tmp[i] = __shfl_up_sync(0xffffffff, P[i], 4);
    if(lane >= 4){ mulmod(P, tmp, tmp2); cpy(P, tmp2); }
    
    for(int i=0;i<8;i++) tmp[i] = __shfl_up_sync(0xffffffff, P[i], 8);
    if(lane >= 8){ mulmod(P, tmp, tmp2); cpy(P, tmp2); }
    
    for(int i=0;i<8;i++) tmp[i] = __shfl_up_sync(0xffffffff, P[i], 16);
    if(lane >= 16){ mulmod(P, tmp, tmp2); cpy(P, tmp2); }
    
    // 2. Inclusive suffix product S
    u32 S[8]; cpy(S, val);
    for(int i=0;i<8;i++) tmp[i] = __shfl_down_sync(0xffffffff, S[i], 1);
    if(lane <= 30){ mulmod(S, tmp, tmp2); cpy(S, tmp2); }
    
    for(int i=0;i<8;i++) tmp[i] = __shfl_down_sync(0xffffffff, S[i], 2);
    if(lane <= 29){ mulmod(S, tmp, tmp2); cpy(S, tmp2); }
    
    for(int i=0;i<8;i++) tmp[i] = __shfl_down_sync(0xffffffff, S[i], 4);
    if(lane <= 27){ mulmod(S, tmp, tmp2); cpy(S, tmp2); }
    
    for(int i=0;i<8;i++) tmp[i] = __shfl_down_sync(0xffffffff, S[i], 8);
    if(lane <= 23){ mulmod(S, tmp, tmp2); cpy(S, tmp2); }
    
    for(int i=0;i<8;i++) tmp[i] = __shfl_down_sync(0xffffffff, S[i], 16);
    if(lane <= 15){ mulmod(S, tmp, tmp2); cpy(S, tmp2); }
    
    // 3. Inverse of total product
    u32 I_total[8];
    if(lane == 31){ invmod(P, I_total); }
    for(int i=0;i<8;i++) I_total[i] = __shfl_sync(0xffffffff, I_total[i], 31);
    
    // 4. Combine: val_inv = P_{lane-1} * S_{lane+1} * I_total
    u32 P_prev[8];
    for(int i=0;i<8;i++) P_prev[i] = __shfl_up_sync(0xffffffff, P[i], 1);
    if(lane == 0) { for(int i=0;i<8;i++) P_prev[i] = (i==0)?1:0; }
    
    u32 S_next[8];
    for(int i=0;i<8;i++) S_next[i] = __shfl_down_sync(0xffffffff, S[i], 1);
    if(lane == 31) { for(int i=0;i<8;i++) S_next[i] = (i==0)?1:0; }
    
    u32 res[8];
    mulmod(P_prev, S_next, res);
    mulmod(res, I_total, val);
}

// ===== Batch-inversion shot-parallel kernel =====
// Pipeline per wave of W shots:
//   1. Parse k1,k2 → comb_mul_jac → store tj/oj Jacobian in s_jxy, Z in s_z
//   2. batch_inv(s_z) → tj.Z inverted → each thread computes tx,ty into s_a,s_b
//   3. Store oj.Z into s_z, oj X,Y into s_jxy → batch_inv(s_z) → oj.Z inverted → ox,oy into s_a,s_b
//   4. Compute den = ox-tx into s_z → batch_inv(s_z) → den inverted → finish per-thread
//
// Wait — step 3 needs s_a,s_b for ox,oy but s_a,s_b still hold tx,ty.
// Solution: use separate arrays for first/second point affine coords.
// s_a = tx, s_b = ty.  After batch 2: s_jxy[0..7] = ox, s_jxy[8..15] = oy
// (overwriting oj Jacobian X,Y which are no longer needed).

#define W 128

__device__ unsigned long long d_global_counter;

__global__ __launch_bounds__(128, 2)
void search_kernel3(u64 start, u64 count, u32* out_cnt, u64* out_list, int max_out, int do_inst){
    __shared__ u64 sq_st[25];          // 200 B
    __shared__ int sq_pt;              // 4 B
    __shared__ unsigned char wbytes[W*64]; // 8192 B
    __shared__ int hard_flag;          // 4 B
    // Total shared memory drastically reduced to ~8.4 KB due to warp_batch_inv
    
    __shared__ unsigned long long s_block_nidx; // 8 B

    int t = threadIdx.x;
    while(true){
        if(t == 0){
            s_block_nidx = atomicAdd((unsigned long long*)&d_global_counter, 1ULL);
        }
        __syncthreads();
        u64 nidx = s_block_nidx;
        if(nidx >= count) break;
        
        u64 nonce = start + nidx;
        if(t==0){
            for(int i=0;i<25;i++) sq_st[i]=d_base_st[i];
            int pt=d_base_pt;
            for(int i=0;i<48;i++){u64 q=((nonce>>i)&1)?d_tx1:d_tx0; feed_x_op_dev(sq_st,pt,q); feed_x_op_dev(sq_st,pt,q);}
            unsigned char* sb=(unsigned char*)sq_st; sb[pt]^=0x1F; sb[135]^=0x80; keccakf(sq_st);
            sq_pt=0; hard_flag=0;
        }
        __syncthreads();
        if(t==0 && do_inst) atomicAdd((unsigned long long*)&d_rej->nonces_launched, 1ULL);

        for(int base_shot=0; base_shot<9024; base_shot+=W){
            if(hard_flag) break;
            int n_this = (9024-base_shot < W) ? (9024-base_shot) : W;

            // Squeeze
            if(t==0){
                unsigned char* sb=(unsigned char*)sq_st;
                int need=n_this*64;
                for(int i=0;i<need;i++){if(sq_pt==136){keccakf(sq_st);sq_pt=0;} wbytes[i]=sb[sq_pt++];}
            }
            __syncthreads();

            Jac tj, oj;
            bool skip = false;
            
            if(t < n_this){
                unsigned char* rb = &wbytes[t*64];
                u32 k1[8], k2[8];
                for(int i=0;i<8;i++) k1[i]=rb[4*i]|(rb[4*i+1]<<8)|(rb[4*i+2]<<16)|(rb[4*i+3]<<24);
                for(int i=0;i<8;i++) k2[i]=rb[32+4*i]|(rb[32+4*i+1]<<8)|(rb[32+4*i+2]<<16)|(rb[32+4*i+3]<<24);

                comb_mul_jac(k1, tj);
                comb_mul_jac(k2, oj);
                if(jacInf(tj) && jacInf(oj)) skip = true;
            } else {
                skip = true;
            }
            
            // Combined Z-inversion: invert tj.Z * oj.Z in one batch, then recover individual inverses.
            // Saves one entire invmod call (~270 heavy ops per warp).
            bool skip_oj = skip || jacInf(oj);
            u32 combined_z[8];
            if(skip) {
                for(int i=0;i<8;i++) combined_z[i] = (i==0)?1:0;
            } else if(skip_oj) {
                cpy(combined_z, tj.Z);  // only need tj.Z inverse
            } else {
                mulmod(tj.Z, oj.Z, combined_z);  // combined = tj.Z * oj.Z
            }
            
            warp_batch_inv(combined_z); // combined_z = 1/(tj.Z * oj.Z)
            
            u32 tx[8], ty[8], ox[8], oy[8];
            if(!skip){
                u32 inv_z1[8];
                if(!skip_oj) {
                    // Recover individual inverses:
                    // 1/tj.Z = combined_inv * oj.Z
                    // 1/oj.Z = combined_inv * tj.Z
                    u32 inv_z2[8];
                    mulmod(combined_z, oj.Z, inv_z1);
                    mulmod(combined_z, tj.Z, inv_z2);
                    
                    // Convert oj to affine
                    u32 zi2[8], zi3[8];
                    sqrmod(inv_z2, zi2);
                    mulmod(zi2, inv_z2, zi3);
                    mulmod(oj.X, zi2, ox);
                    mulmod(oj.Y, zi3, oy);
                } else {
                    cpy(inv_z1, combined_z); // combined was just tj.Z
                    skip = true; // no oj to process
                }
                
                // Convert tj to affine
                u32 zi2[8], zi3[8];
                sqrmod(inv_z1, zi2);
                mulmod(zi2, inv_z1, zi3);
                mulmod(tj.X, zi2, tx);
                mulmod(tj.Y, zi3, ty);
                
                if(!skip_oj) {
                    skip = false; // restore
                    if(eq(tx, ox)) skip = true;
                    if(isZero(tx) && isZero(ty)) skip = true;
                    if(isZero(ox) && isZero(oy)) skip = true;
                }
            }
            
            int local_hard_flag = 0;
            if(!skip){
                u32 dx[8]; submod_p(tx, ox, dx);
                if(do_inst){
                    int rej = check_gcd_factor_inst(dx, 0);
                    if(rej){ local_hard_flag = 1; atomicAdd((unsigned long long*)&d_rej->first_dx, 1ULL); }
                } else {
                    if(!check_gcd_factor(dx)) local_hard_flag = 1;
                }
            }
            
            if(local_hard_flag) hard_flag = 1;
            __syncthreads();
            if(hard_flag) break;
            
            u32 den[8];
            if(skip) { for(int i=0;i<8;i++) den[i] = (i==0)?1:0; }
            else { submod_p(ox, tx, den); }
            
            warp_batch_inv(den); // den is now 1/(ox-tx)
            
            if(!skip){
                u32 num[8]; submod_p(oy, ty, num);
                u32 lambda[8]; mulmod(num, den, lambda);
                u32 ex[8]; sqrmod(lambda, ex);
                submod(ex, tx); submod(ex, ox);
                u32 c[8]; submod_p(ox, ex, c);
                
                if(do_inst){
                    int rej = check_gcd_factor_inst(c, 1);
                    if(rej){ local_hard_flag = 1; atomicAdd((unsigned long long*)&d_rej->first_c, 1ULL); }
                } else {
                    if(!check_gcd_factor(c)) local_hard_flag = 1;
                }
            }
            
            if(local_hard_flag) hard_flag = 1;
            __syncthreads();
        }

        if(t==0 && !hard_flag){
            if(do_inst) atomicAdd((unsigned long long*)&d_rej->nonces_clean, 1ULL);
            u32 pos=atomicAdd(out_cnt,1u);
            if(pos<(u32)max_out) out_list[pos]=nonce;
        }
        __syncthreads();
    }
}

// Probe kernel
__global__ void probe_kernel3(u64 nonce, u32* out){
    u64 sq_st[25]; for(int i=0;i<25;i++) sq_st[i]=d_base_st[i];
    int pt=d_base_pt;
    for(int i=0;i<48;i++){u64 q=((nonce>>i)&1)?d_tx1:d_tx0; feed_x_op_dev(sq_st,pt,q); feed_x_op_dev(sq_st,pt,q);}
    unsigned char* sb=(unsigned char*)sq_st; sb[pt]^=0x1F; sb[135]^=0x80; keccakf(sq_st);
    int sq_pt=0;
    uint8_t rb[64];
    for(int i=0;i<64;i++){if(sq_pt==136){keccakf(sq_st);sq_pt=0;} rb[i]=sb[sq_pt++];}
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
        probe_kernel3<<<1,1>>>(probe,dout);
        cudaError_t e=cudaDeviceSynchronize(); if(e){printf("probe err %s\n",cudaGetErrorString(e));return 1;}
        u32 h[16]; cudaMemcpy(h,dout,16*4,cudaMemcpyDeviceToHost);
        bool ok1=true,ok2=true; for(int i=0;i<8;i++){if(h[i]!=pk1[i])ok1=false; if(h[8+i]!=pk2[i])ok2=false;}
        printf("probe nonce=%llu k1:%s k2:%s\n",(unsigned long long)probe, ok1?"OK":"MISMATCH", ok2?"OK":"MISMATCH");
        return 0;
    }

    u32* dcnt; cudaMalloc(&dcnt,4); cudaMemset(dcnt,0,4);
    const int MAXOUT=4096; u64* dlist; cudaMalloc(&dlist,MAXOUT*8);
    int blocks = getenv("BLOCKS")? atoi(getenv("BLOCKS")) : 72;
    unsigned long long zero_counter = 0;
    cudaMemcpyToSymbol(d_global_counter, &zero_counter, 8);
    bool do_inst = getenv("INSTRUMENT")!=NULL;
    RejCounters* d_rc=NULL; RejCounters h_rc={};
    if(do_inst){
        cudaMalloc(&d_rc, sizeof(RejCounters)); cudaMemset(d_rc,0,sizeof(RejCounters));
        cudaMemcpyToSymbol(d_rej, &d_rc, sizeof(d_rc));
    }
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1); cudaEventRecord(t0);
    search_kernel3<<<blocks,W>>>(start,count,dcnt,dlist,MAXOUT, do_inst?1:0);
    cudaError_t e=cudaDeviceSynchronize(); if(e){printf("search err %s\n",cudaGetErrorString(e));return 1;}
    cudaEventRecord(t1); cudaEventSynchronize(t1); float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    u32 cnt; cudaMemcpy(&cnt,dcnt,4,cudaMemcpyDeviceToHost);
    u64 list[MAXOUT]; cudaMemcpy(list,dlist,(cnt<MAXOUT?cnt:MAXOUT)*8,cudaMemcpyDeviceToHost);
    for(u32 i=0;i<cnt && i<MAXOUT;i++) printf("CLEAN nonce=%llu\n",(unsigned long long)list[i]);
    printf("scanned %llu in %.2fs (%llu nonce/s); clean=%d [kernel3 batch-inv x2 + addchain-inv]\n",
        (unsigned long long)count, ms/1000.0, (u64)(count/(ms/1000.0)), cnt);
    if(do_inst && d_rc){
        cudaMemcpy(&h_rc, d_rc, sizeof(RejCounters), cudaMemcpyDeviceToHost);
        u64 dx_tot = h_rc.dx_zero+h_rc.dx_noconv+h_rc.dx_width;
        u64 c_tot  = h_rc.c_zero+h_rc.c_noconv+h_rc.c_width;
        u64 all_hard = dx_tot+c_tot;
        printf("\n=== Rejection Profile ===\n");
        printf("nonces: %llu launched, %llu clean (%.4f%%)\n",
            (unsigned long long)h_rc.nonces_launched, (unsigned long long)h_rc.nonces_clean,
            h_rc.nonces_launched ? 100.0*h_rc.nonces_clean/h_rc.nonces_launched : 0.0);
        printf("hard: %llu (dx=%llu, c=%llu)\n", (unsigned long long)all_hard,
            (unsigned long long)dx_tot, (unsigned long long)c_tot);
        if(all_hard){
            printf("  dx: zero=%llu(%.1f%%) noconv=%llu(%.1f%%) width=%llu(%.1f%%)\n",
                (unsigned long long)h_rc.dx_zero,  100.0*h_rc.dx_zero/all_hard,
                (unsigned long long)h_rc.dx_noconv,100.0*h_rc.dx_noconv/all_hard,
                (unsigned long long)h_rc.dx_width, 100.0*h_rc.dx_width/all_hard);
            printf("   c: zero=%llu(%.1f%%) noconv=%llu(%.1f%%) width=%llu(%.1f%%)\n",
                (unsigned long long)h_rc.c_zero,  100.0*h_rc.c_zero/all_hard,
                (unsigned long long)h_rc.c_noconv,100.0*h_rc.c_noconv/all_hard,
                (unsigned long long)h_rc.c_width, 100.0*h_rc.c_width/all_hard);
        }
        printf("first rejection: dx=%llu c=%llu\n",
            (unsigned long long)h_rc.first_dx, (unsigned long long)h_rc.first_c);
        printf("=========================\n");
    }
    return 0;
}
