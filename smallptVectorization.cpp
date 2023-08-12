//g++ -std=c++11 -O3 -fopenmp -march=native -mtune=native -mavx -mfma smallptVectorization.cpp -o smallptVectorization

#include <cstdint>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <limits>

struct Vec {
    double x, y, z; //position, also color (r,g,b)
    Vec(double x_=0, double y_=0, double z_=0) { 
        x = x_; y = y_; z = z_; 
    }
    Vec operator+(const Vec &b) const {
        return Vec(x+b.x, y+b.y, z+b.z); 
    }
    Vec operator-(const Vec &b) const {
        return Vec(x-b.x, y-b.y, z-b.z); 
    }
    Vec operator*(double b) const {
        return Vec(x*b, y*b, z*b); 
    }
    Vec mult(const Vec &b) const {
        return Vec(x*b.x, y*b.y, z*b.z); 
    }
    //Vectorized norm function
    Vec& norm() {
        //Create a vector with 4 elements inside a 256 bit register
        __m256d v = _mm256_set_pd(z, y, x, 0);
        
        //Square each element of the register
        v = _mm256_mul_pd(v, v);

        //Seperate the vector in half
        __m128d v_low  = _mm256_castpd256_pd128(v);
        __m128d v_high = _mm256_extractf128_pd(v, 1);

        //Add the lower and higher halves of the register together
        v_low  = _mm_add_pd(v_low, v_high);

        //Unpack the lower half of the register to the high half of another register, 
        //duplicate the values in the lower half, and add them together
        v_high = _mm_unpackhi_pd(v_low, v_low);
        v_low  = _mm_add_pd(v_low, v_high);

        //Convert the sum from a 128-bit register to a double value
        double sum = _mm_cvtsd_f64(v_low);

        //Compute the inverse square root of the sum
        double norm = 1.0 / sqrt(sum);

        //Multiply each element of the original register by the inverse square root
        x *= norm;
        y *= norm;
        z *= norm;

        //Return a the Vec
        return *this;
    }

    double dot(const Vec &b) const {
        return x*b.x + y*b.y + z*b.z; 
    } 
    Vec operator%(Vec&b) {
        return Vec(y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x);
    }
};

//Ray struct with origin and direction
struct Ray {
    Vec o, d; 
    Ray(Vec o_, Vec d_) : o(o_), d(d_) {} 
};

//Material reflection types (diffuse,specular,refractive)
enum Refl_t { 
    DIFF,
    SPEC,
    REFR
};  

//Sphere struct with radius, position, emission, color, and reflection type
struct Sphere {
    double rad;   //radius
    Vec p, e, c;  //position, emission, color
    Refl_t refl;  //reflection type
    Sphere(double rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_) :
        rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}
    //Computes the intersection of the ray with the sphere
    double intersect(const Ray &r) const {
        Vec op = p - r.o; 
        double t, eps = 1e-4, b = op.dot(r.d), 
        det = b*b - op.dot(op) + rad*rad;
        if (det < 0) return 0; 
        else det = sqrt(det);
        return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
    }
};

//Scene with spheres
Sphere spheres[] = {
    //radius, position, emission, color, material
    Sphere(1e5, Vec(1e5+1, 40.8, 81.6), Vec(), Vec(.75,.25,.25), DIFF), //Left
    Sphere(1e5, Vec(-1e5+99, 40.8, 81.6), Vec(), Vec(.25,.25,.75), DIFF), //Right
    Sphere(1e5, Vec(50, 40.8, 1e5), Vec(), Vec(.75,.75,.75), DIFF), //Back
    Sphere(1e5, Vec(50, 40.8, -1e5+170), Vec(), Vec(), DIFF), //Front
    Sphere(1e5, Vec(50, 1e5, 81.6), Vec(), Vec(.75,.75,.75), DIFF), //Bottom
    Sphere(1e5, Vec(50, -1e5+81.6, 81.6), Vec(), Vec(.75,.75,.75), DIFF), //Top
    Sphere(16.5, Vec(27, 16.5, 47), Vec(), Vec(1,1,1)*.999, SPEC), //Mirror
    Sphere(16.5, Vec(73, 16.5, 78), Vec(), Vec(1,1,1)*.999, REFR), //Glass
    Sphere(600, Vec(50, 681.6-.27, 81.6), Vec(12,12,12), Vec(), DIFF) //Light
};

inline double clamp(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
inline int toInt(double x) { return int(pow(clamp(x), 1/2.2) * 255 + .5); }

inline bool intersect(const Ray &r, double &t, int &id) {
    double n = sizeof(spheres)/sizeof(Sphere), d, inf = t = 1e20;
    for (int i = int(n); i--;) if ((d = spheres[i].intersect(r)) && d < t) { t = d; id = i; }
    return t < inf;
}

Vec radiance(const Ray& r, int depth, unsigned short* Xi) {
    double t; //distance to intersection
    int id = 0; //id of intersected object
    if (!intersect(r, t, id)) return Vec(); //if miss, return black
    const Sphere& obj = spheres[id]; //the hit object
    Vec x = r.o + r.d * t, n = (x - obj.p).norm(), nl = n.dot(r.d) < 0 ? n : n * -1, f = obj.c;
    double p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z; //max refl
    if (++depth > 5) {
        if (erand48(Xi) < p) f = f * (1 / p); else return obj.e; //Russian roulette
    }
    if (obj.refl == DIFF) { //Ideal diffuse reflection
        double r1 = 2 * M_PI * erand48(Xi), r2 = erand48(Xi), r2s = sqrt(r2);
        Vec w = nl, u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm(), v = w % u;
        Vec d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();
        return obj.e + f.mult(radiance(Ray(x, d), depth, Xi));
    }
    else if (obj.refl == SPEC) //Ideal SPECULAR reflection
        return obj.e + f.mult(radiance(Ray(x, r.d - n * 2 * n.dot(r.d)), depth, Xi));
    Ray reflRay(x, r.d - n * 2 * n.dot(r.d)); //Ideal dielectric REFRACTION
    bool into = n.dot(nl) > 0; //Ray from outside going in?
    double nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = r.d.dot(nl), cos2t;
    if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) < 0) //Total internal reflection
        return obj.e + f.mult(radiance(reflRay, depth, Xi));
    Vec tdir = (r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).norm();
    double a = nt - nc, b = nt + nc, R0 = a * a / (b * b), c = 1 - (into ? -ddn : tdir.dot(n));
    double Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = .25 + .5 * Re, RP = Re / P, TP = Tr / (1 - P);
    return obj.e + f.mult(depth > 2 ? (erand48(Xi) < P ? //Russian roulette
        radiance(reflRay, depth, Xi) * RP : radiance(Ray(x, tdir), depth, Xi) * TP) :
        radiance(reflRay, depth, Xi) * Re + radiance(Ray(x, tdir), depth, Xi) * Tr);
}

int main(int argc, char *argv[]) {
    int w = 1024;
    int h = 768;
    int samps = argc == 2 ? atoi(argv[1]) / 4 : 1; //# samples
    
    //Setup camera
    Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm());
    Vec cx = Vec(w * 0.5135 / h);
    Vec cy = (cx % cam.d).norm() * 0.5135;
    
    //Variable for each ray
    Vec r; 

    //The final canvas
    Vec *c = new Vec[w * h];

    #pragma omp parallel for simd schedule(dynamic) private(r) //Промяната от static на dynamic ускори леко процеса и оправи проблема с изписването на процентите
    //Loop over image rows
    for (int y = 0; y < h; y++) {
        fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samps * 4, 100.0 * y / (h - 1));
        
        //Loop over columns
        for (unsigned short x = 0, Xi[3] = {0, 0, y * y * y}; x < w; x++) {
            //Loop over subpixel rows
            for (int sy = 0, i = (h - y - 1) * w + x; sy < 2; sy++) {
                //Loop over subpixel columns
                for (int sx = 0; sx < 2; sx++, r = Vec()) {
                    //Loop over samples for each subpixel
                    for (int s = 0; s < samps; s++) {

                        double r1 = 2 * erand48(Xi);
                        double dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                        
                        double r2 = 2 * erand48(Xi);
                        double dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
                        
                        Vec d = cx * (((sx + 0.5 + dx) / 2 + x) / w - 0.5) +
                                cy * (((sy + 0.5 + dy) / 2 + y) / h - 0.5) + cam.d;
                        
                        //Calculate radiance from all the samples
                        r = r + radiance(Ray(cam.o + d * 140, d.norm()), 0, Xi) * (1.0 / samps);
                    }
                    
                    //Store subpixel in canvas
                    c[i] = c[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z)) * 0.25;
                }
            }
        }
    }
    
    //Write image to PPM file
    FILE *f = fopen("image.ppm", "w");
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i = 0; i < w * h; i++) {
        fprintf(f, "%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
    }
}