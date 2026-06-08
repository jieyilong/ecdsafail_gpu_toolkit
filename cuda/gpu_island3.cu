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
__device__ __forceinline__ void mul_raw(const u32 a[8], const u32 b[8], u32 prod[16]){
    u64 r[16]; for(int i=0;i<16;i++) r[i]=0;
    for(int i=0;i<8;i++){
        u64 carry=0;
        for(int j=0;j<8;j++){ u64 cur = r[i+j]+(u64)a[i]*b[j]+carry; r[i+j]=(u32)cur; carry=cur>>32; }
        r[i+8]+=carry;
    }
    for(int i=0;i<16;i++) prod[i]=(u32)r[i];
}
__device__ __forceinline__ void reduce512(u32 t[16], u32 out[8]){
    u32 w[16]; for(int i=0;i<16;i++) w[i]=t[i];
    for(int iter=0;iter<4;iter++){
        bool hz=true; for(int i=8;i<16;i++) if(w[i]){hz=false;break;}
        if(hz) break;
        u64 acc[10]; for(int i=0;i<10;i++) acc[i]=0;
        for(int i=0;i<8;i++) acc[i]=w[i];
        u64 carry=0;
        for(int i=0;i<8;i++){ u64 cur=acc[i]+(u64)w[8+i]*C_LOW+carry; acc[i]=(u32)cur; carry=cur>>32; }
        acc[8]+=carry; carry=0;
        for(int i=0;i<8;i++){ u64 cur=acc[i+1]+(u64)w[8+i]+carry; acc[i+1]=(u32)cur; carry=cur>>32; }
        acc[9]+=carry;
        for(int i=0;i<16;i++) w[i]=(i<10)?(u32)acc[i]:0;
    }
    for(int i=0;i<8;i++) out[i]=w[i];
    if(geP(out)) subP(out); if(geP(out)) subP(out);
}
__device__ __forceinline__ void mulmod(const u32 a[8], const u32 b[8], u32 out[8]){
    u32 prod[16]; mul_raw(a,b,prod); reduce512(prod,out);
}
__device__ __forceinline__ void sqrmod(const u32 a[8], u32 out[8]){ mulmod(a,a,out); }
__device__ __forceinline__ void cpy(u32 d[8], const u32 s[8]){ for(int i=0;i<8;i++) d[i]=s[i]; }
__device__ __forceinline__ bool isZero(const u32 a[8]){ for(int i=0;i<8;i++) if(a[i]) return false; return true; }
__device__ __forceinline__ bool eq(const u32 a[8], const u32 b[8]){ for(int i=0;i<8;i++) if(a[i]!=b[i]) return false; return true; }

__device__ __constant__ u32 PM2[8]={0xFFFFFC2D,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF};
__device__ void invmod(const u32 a[8], u32 out[8]){
    u32 r[8]={1,0,0,0,0,0,0,0}; u32 base[8]; cpy(base,a);
    for(int i=0;i<256;i++){
        if((PM2[i>>5]>>(i&31))&1){ u32 t[8]; mulmod(r,base,t); cpy(r,t); }
        u32 t[8]; sqrmod(base,t); cpy(base,t);
    }
    cpy(out,r);
}

struct Jac { u32 X[8],Y[8],Z[8]; };
__device__ __forceinline__ bool jacInf(const Jac&p){ return isZero(p.Z); }
__device__ void jacDouble(const Jac&p, Jac&r){
    if(jacInf(p)||isZero(p.Y)){ for(int i=0;i<8;i++){r.X[i]=0;r.Y[i]=0;r.Z[i]=0;} return; }
    u32 YY[8],S[8],M[8],t[8],t2[8];
    sqrmod(p.Y,YY);
    mulmod(p.X,YY,S); u32 four[8]={4,0,0,0,0,0,0,0}; mulmod(S,four,S);
    sqrmod(p.X,M); u32 three[8]={3,0,0,0,0,0,0,0}; mulmod(M,three,M);
    sqrmod(M,r.X); submod(r.X,S); submod(r.X,S);
    sqrmod(YY,t2); u32 eight[8]={8,0,0,0,0,0,0,0}; mulmod(t2,eight,t2);
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

// ===== Montgomery batch inversion =====
// Inverts N values in vals[0..N-1][8] in-place.
// pf[N][8] and orig[N][8] are scratch.  N <= blockDim.x.
// All N threads call this; t == threadIdx.x.
__device__ void batch_inv(u32 vals[][8], u32 pf[][8], u32 orig[][8], int N, int t){
    // Phase 0: backup originals
    if(t < N) cpy(orig[t], vals[t]);
    __syncthreads();
    // Phase 1: forward prefix product + single Fermat inverse (t==0 serial)
    if(t == 0){
        cpy(pf[0], orig[0]);
        for(int i=1;i<N;i++){u32 tmp[8]; mulmod(pf[i-1],orig[i],tmp); cpy(pf[i],tmp);}
        u32 inv_all[8]; invmod(pf[N-1], inv_all);
        cpy(pf[0], inv_all); // broadcast via pf[0]
    }
    __syncthreads();
    // Phase 2: backward pass to recover individual inverses (t==0 serial)
    if(t == 0){
        u32 running[8]; cpy(running, pf[0]); // = 1/(product of all)
        for(int i=N-1;i>=1;i--){
            u32 inv_i[8]; mulmod(running, pf[i-1], inv_i);
            cpy(vals[i], inv_i);
            u32 tmp[8]; mulmod(running, orig[i], tmp); cpy(running, tmp);
        }
        cpy(vals[0], running);
    }
    __syncthreads();
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

__global__ void search_kernel3(u64 start, u64 count, u32* out_cnt, u64* out_list, int max_out){
    __shared__ u64 sq_st[25];          // 200 B
    __shared__ int sq_pt;              // 4 B
    __shared__ unsigned char wbytes[W*64]; // 8192 B
    __shared__ int hard_flag;          // 4 B
    // Batch inversion pipeline:
    __shared__ u32 s_z[W][8];          // 4096 B  Z values (reused for tj.Z, oj.Z, den)
    __shared__ u32 s_a[W][8];          // 4096 B  tx (point 1 affine x)
    __shared__ u32 s_b[W][8];          // 4096 B  ty (point 1 affine y)
    __shared__ u32 s_jxy[W][16];       // 8192 B  Jacobian X[8]||Y[8] then affine ox[8]||oy[8]
    __shared__ u32 s_pf[W][8];         // 4096 B  batch-inv scratch (prefix)
    __shared__ u32 s_or[W][8];         // 4096 B  batch-inv scratch (orig backup)
    __shared__ int s_skip[W];          // 512 B
    // Total shared: ~41,916 B ≈ 41 KB

    int t = threadIdx.x;
    for(u64 nidx = blockIdx.x; nidx < count; nidx += gridDim.x){
        u64 nonce = start + nidx;
        if(t==0){
            for(int i=0;i<25;i++) sq_st[i]=d_base_st[i];
            int pt=d_base_pt;
            for(int i=0;i<48;i++){u64 q=((nonce>>i)&1)?d_tx1:d_tx0; feed_x_op_dev(sq_st,pt,q); feed_x_op_dev(sq_st,pt,q);}
            unsigned char* sb=(unsigned char*)sq_st; sb[pt]^=0x1F; sb[135]^=0x80; keccakf(sq_st);
            sq_pt=0; hard_flag=0;
        }
        __syncthreads();

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

            // Each thread: parse k1,k2, compute comb_mul_jac for BOTH points,
            // store Jacobian X,Y in s_jxy, Z in s_z.  First point: s_jxy = tj.X||tj.Y, s_z = tj.Z.
            // Second point: we store oj into registers temporarily, then after batch 1 we'll
            // put oj.X||oj.Y into s_jxy and oj.Z into s_z.
            if(t < n_this){
                unsigned char* rb = &wbytes[t*64];
                u32 k1[8], k2[8];
                for(int i=0;i<8;i++) k1[i]=rb[4*i]|(rb[4*i+1]<<8)|(rb[4*i+2]<<16)|(rb[4*i+3]<<24);
                for(int i=0;i<8;i++) k2[i]=rb[32+4*i]|(rb[32+4*i+1]<<8)|(rb[32+4*i+2]<<16)|(rb[32+4*i+3]<<24);

                Jac tj, oj;
                comb_mul_jac(k1, tj);
                comb_mul_jac(k2, oj);

                // Store tj into shared
                for(int i=0;i<8;i++) s_jxy[t][i]   = tj.X[i];
                for(int i=0;i<8;i++) s_jxy[t][8+i] = tj.Y[i];
                cpy(s_z[t], tj.Z);

                // Detect early skip: both points at infinity
                s_skip[t] = (jacInf(tj) && jacInf(oj)) ? 1 : 0;

                // Store oj Jacobian in per-thread registers until after batch 1
                // (We'll write them to shared memory between batch 1 and batch 2)
                // Store in s_or temporarily — BUT s_or is batch_inv scratch.
                // s_or is only used inside batch_inv. After batch_inv returns, s_or is free.
                // So we can write oj data to s_or AFTER batch_inv finishes.
                // For now, keep oj in registers — we'll store it between batch 1 and 2.
                // This means we need: oj.X[8], oj.Y[8], oj.Z[8] in registers.
                // That's 24 extra registers per thread. On top of comb_mul_jac's register pressure.
                // To avoid register bloat, let's store oj into s_or/s_pf AFTER batch 1's
                // batch_inv call finishes (s_or and s_pf are then free).

                // Actually, we need to store oj.Z into s_z[t] AFTER batch 1 reads s_z[t]
                // (which had tj.Z). And oj.X,Y into s_jxy[t] AFTER batch 1 reads s_jxy[t]
                // (which had tj.X,Y). So the sequence is:
                // 1. Store tj into s_jxy/s_z, compute oj in registers
                // 2. batch_inv(s_z) — inverts tj.Z
                // 3. Compute tx,ty from s_jxy (tj.X,Y) and s_z (1/tj.Z)
                // 4. Now safe to overwrite: store oj into s_jxy/s_z
                // 5. batch_inv(s_z) — inverts oj.Z
                // 6. Compute ox,oy from s_jxy (oj.X,Y) and s_z (1/oj.Z), store in s_jxy
                // 7. Compute den into s_z, batch_inv(s_z)
                // 8. Finish with lambda = num * (1/den), check_gcd_factor
            } else {
                s_skip[t] = 1;
            }
            __syncthreads();

            // --- Batch 1: invert tj.Z ---
            batch_inv(s_z, s_pf, s_or, n_this, t);
            // s_z[t] now = 1/tj.Z

            // Compute affine tx,ty for point 1 and store in s_a, s_b
            if(t < n_this && !s_skip[t]){
                u32 zi2[8], zi3[8];
                sqrmod(s_z[t], zi2);
                mulmod(zi2, s_z[t], zi3);
                mulmod(s_jxy[t], zi2, s_a[t]);          // tx = tj.X * zi2
                mulmod(&s_jxy[t][8], zi3, s_b[t]);      // ty = tj.Y * zi3
            }
            __syncthreads();

            // Now store oj Jacobian into shared (overwriting data batch 1 already consumed)
            // We need oj in registers. But wait — we computed oj above but the register
            // allocation happened inside the `if(t < n_this)` block. The compiler may have
            // already spilled oj. We need to keep oj alive.
            //
            // To avoid this register pressure problem, let's RECOMPUTE oj from k2.
            // comb_mul_jac(k2, oj) is ~200 mulmods — cheaper than 24 registers held across
            // a sync point and a batch_inv call (which the compiler can't optimize across).
            // Actually on second thought, let's try keeping it in registers. The compiler
            // should handle this since it's all within the same if-block.
            //
            // BUT the __syncthreads() inside batch_inv means the compiler MUST spill any
            // register-held values that cross the sync. So oj WILL be spilled to local memory
            // (which is slow on GPU — physical local memory = L1 cache pressure).
            //
            // Decision: recompute oj after batch 1. Cost: ~200 mulmods. Benefit: no spills.
            // The 200 mulmods are cheap compared to the 768 Fermat invmod savings.

            if(t < n_this){
                unsigned char* rb = &wbytes[t*64];
                u32 k2[8];
                for(int i=0;i<8;i++) k2[i]=rb[32+4*i]|(rb[32+4*i+1]<<8)|(rb[32+4*i+2]<<16)|(rb[32+4*i+3]<<24);
                Jac oj; comb_mul_jac(k2, oj);

                // Store oj into shared
                for(int i=0;i<8;i++) s_jxy[t][i]   = oj.X[i];
                for(int i=0;i<8;i++) s_jxy[t][8+i] = oj.Y[i];
                cpy(s_z[t], oj.Z);
                if(jacInf(oj)) s_skip[t] = 1;
            }
            __syncthreads();

            // --- Batch 2: invert oj.Z ---
            batch_inv(s_z, s_pf, s_or, n_this, t);

            // Compute affine ox,oy and store in s_jxy (overwriting oj Jacobian X,Y)
            // Also detect skip conditions that need both affine coords
            if(t < n_this && !s_skip[t]){
                u32 zi2[8], zi3[8];
                sqrmod(s_z[t], zi2);
                mulmod(zi2, s_z[t], zi3);
                mulmod(s_jxy[t], zi2, s_jxy[t]);           // ox → s_jxy[0..7]
                mulmod(&s_jxy[t][8], zi3, &s_jxy[t][8]);   // oy → s_jxy[8..15]

                // Skip checks (need both affine coords now)
                if(eq(s_a[t], s_jxy[t]))           s_skip[t] = 1; // tx == ox
                if(isZero(s_a[t])&&isZero(s_b[t]))  s_skip[t] = 1; // tx,ty = 0
                if(isZero(s_jxy[t])&&isZero(&s_jxy[t][8])) s_skip[t] = 1; // ox,oy = 0
            }
            __syncthreads();

            // Compute den = ox - tx into s_z (skip threads get den=1 for batch safety)
            if(t < n_this && !s_skip[t]){
                submod_p(s_jxy[t], s_a[t], s_z[t]); // den = ox - tx
            } else if(t < n_this) {
                for(int i=0;i<8;i++) s_z[t][i] = (i==0)?1u:0u; // den = 1 for skipped
            }
            __syncthreads();

            // --- Batch 3: invert den ---
            batch_inv(s_z, s_pf, s_or, n_this, t);

            // Finish: each thread computes lambda, ex, dx, c, check_gcd_factor
            if(t < n_this && !s_skip[t]){
                // num = oy - ty
                u32 num[8]; submod_p(&s_jxy[t][8], s_b[t], num);
                // lambda = num * (1/den)
                u32 lambda[8]; mulmod(num, s_z[t], lambda);
                // ex = lambda^2 - tx - ox
                u32 ex[8]; sqrmod(lambda, ex);
                submod(ex, s_a[t]);    // ex -= tx
                submod(ex, s_jxy[t]);  // ex -= ox
                // dx = tx - ox, c = ox - ex
                u32 dx[8]; submod_p(s_a[t], s_jxy[t], dx);
                u32 c[8];  submod_p(s_jxy[t], ex, c);

                if(!check_gcd_factor(dx) || !check_gcd_factor(c))
                    hard_flag = 1;
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
    int blocks = getenv("BLOCKS")? atoi(getenv("BLOCKS")) : 512;
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1); cudaEventRecord(t0);
    search_kernel3<<<blocks,W>>>(start,count,dcnt,dlist,MAXOUT);
    cudaError_t e=cudaDeviceSynchronize(); if(e){printf("search err %s\n",cudaGetErrorString(e));return 1;}
    cudaEventRecord(t1); cudaEventSynchronize(t1); float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    u32 cnt; cudaMemcpy(&cnt,dcnt,4,cudaMemcpyDeviceToHost);
    u64 list[MAXOUT]; cudaMemcpy(list,dlist,(cnt<MAXOUT?cnt:MAXOUT)*8,cudaMemcpyDeviceToHost);
    for(u32 i=0;i<cnt && i<MAXOUT;i++) printf("CLEAN nonce=%llu\n",(unsigned long long)list[i]);
    printf("scanned %llu in %.2fs (%.0f nonce/s); clean=%u [kernel3 batch-inv x3]\n",
        (unsigned long long)count, ms/1000.0, count/(ms/1000.0), cnt);
    return 0;
}
