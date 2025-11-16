#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct GlobalConstants
{
    public:
        const double dx = 0.25;
        const double dt = 1.0;
        
        double Du = 0.16;
        double Dv = 0.08;

      // Weird continuousish constantly changing but periodic
      //   double F = 0.028910;
      //   double k = 0.053719;

         double F = 0.03;
         double k = 0.055;

};