using namespace metal;
struct Soft {
    ulong v;
    constexpr Soft() : v(0) {}
    constexpr Soft(ulong raw, bool) : v(raw) {}
    Soft(int i) : v(__softfloat64_cvt_i64_to_f64(i,0)) {}
    explicit operator int() const { return int(__softfloat64_cvt_f64_to_i64(v,3)); }
};
inline Soft operator+(Soft a, Soft b) {return Soft(__softfloat64_fadd(a.v,b.v,0),true);}
inline Soft operator-(Soft a, Soft b) {return Soft(__softfloat64_fsub(a.v,b.v,0),true);}
inline Soft operator*(Soft a, Soft b) {return Soft(__softfloat64_fmul(a.v,b.v,0),true);}
inline Soft operator/(Soft a, Soft b) {return Soft(__softfloat64_fdiv(a.v,b.v,0),true);}
constexpr Soft operator-(Soft a) {return Soft(a.v ^ (1ul<<63),true);}
inline bool operator==(Soft a, Soft b) {return __softfloat64_feq(a.v,b.v);}
inline bool operator!=(Soft a, Soft b) {return !(a==b);}
inline bool operator<(Soft a, Soft b) {return __softfloat64_flt(a.v,b.v);}
inline bool operator>(Soft a, Soft b) {return b<a;}
inline bool operator<=(Soft a, Soft b) {return __softfloat64_fle(a.v,b.v);}
inline bool operator>=(Soft a, Soft b) {return b<=a;}
inline void operator+=(thread Soft& a, Soft b) {a=a+b;}
inline void operator-=(thread Soft& a, Soft b) {a=a-b;}
inline void operator*=(thread Soft& a, Soft b) {a=a*b;}
inline Soft fabs(Soft a) {return Soft(a.v & ~(1ul<<63),true);}
inline Soft sqrt(Soft a) {return Soft(__softfloat64_fsqrt(a.v,0),true);}
inline Soft copysign(Soft a, Soft b) {return Soft(fabs(a).v | (b.v & (1ul<<63)),true);}
#define GET_HIGH_WORD(h,x) ((h)=uint((x).v >> 32))
#define GET_LOW_WORD(l,x) ((l)=uint((x).v))
#define SET_HIGH_WORD(x,h) ((x).v=(ulong(uint(h))<<32)|uint((x).v))
#define SET_LOW_WORD(x,l) ((x).v=((x).v & 0xffffffff00000000ul)|uint(l))
#define INSERT_WORDS(x,h,l) ((x).v=(ulong(uint(h))<<32)|uint(l))
#define EXTRACT_WORDS(h,l,x) do {GET_HIGH_WORD(h,x);GET_LOW_WORD(l,x);} while(0)
#define STRICT_ASSIGN(t,x,v) ((x)=(v))
inline void operator/=(thread Soft& a, Soft b) {a=a/b;}
