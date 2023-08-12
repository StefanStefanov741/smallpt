//mpic++ -o smallptMPI2 smallptMPI2.cpp
//mpiexec -n 6 ./smallptMPI2

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <iostream>
#include <fstream>

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
  Vec& norm() {
    return *this = *this * (1/sqrt(x*x + y*y + z*z)); 
  }
  double dot(const Vec &b) const {
    return x*b.x + y*b.y + z*b.z; 
  } 
  Vec operator%(Vec&b) {
    return Vec(y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x);
  }
  double length() const {
    return std::sqrt(x*x + y*y + z*z);
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

  //MPI initialization
  int ierr;
  int nodeID,nodesCount;

  ierr = MPI_Init(&argc,&argv);
  ierr = MPI_Comm_size(MPI_COMM_WORLD,&nodesCount);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&nodeID);

  int rowsPerNode = h/nodesCount;

  //Variable for each process
  Vec r; 

  //Allocate memory for the portion of the canvas that this process will process
  Vec* localC = new Vec[w * h];
  for(int y=nodeID;y<h;y+=nodesCount){
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
          //Store subpixel in local canvas
          localC[i] = localC[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z)) * 0.25;
        }
      }
    }
  }


  if (nodeID == 0) {
    //Allocate memory to store the final canvas on node 0
    Vec *final_canvas = new Vec[w * h];

    //Copy the rendered local canvas rows from node 0 to the final canvas
    for (int y = nodeID; y < h; y++) {
        int rowStart = y * w;
        for (int k = 0; k < w; k++) {
          if (localC[rowStart + k].length() != 0) {
              memcpy(final_canvas + rowStart, localC + rowStart, w * sizeof(Vec));
              break;
          }
        }
    }

    Vec *temp_pixels = new Vec[w * h];
    //Receive pixel data from the other nodes and store them in the final canvas
    for (int i = 1; i < nodesCount; i++) {
        Vec *temp_pixels = new Vec[w * h];
        MPI_Recv(temp_pixels, w * h * 3, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int j = i; j < h; j++) {
            int rowStart = j * w;
            for (int k = 0; k < w; k++) {
                if (temp_pixels[rowStart + k].length() != 0) {
                    memcpy(final_canvas + rowStart, temp_pixels + rowStart, w * sizeof(Vec));
                    break;
                }
            }
        }
    }
    delete[] temp_pixels;

    //Write final canvas
    FILE* f = fopen("image.ppm", "w");
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        int i = y * w + x;
        fprintf(f, "%d %d %d ", toInt(final_canvas[i].x), toInt(final_canvas[i].y), toInt(final_canvas[i].z));
      }
    }
    fclose(f);

    delete[] final_canvas;
  } else {
    //Flatten the local canvas into a 1D array of pixels and send it to node 0
    Vec *local_pixels = new Vec[w * h];
    for (int y = 0; y < h; y++) {
        int rowStart = y * w;
        memcpy(local_pixels + rowStart, localC + rowStart, w * sizeof(Vec));
    }

    MPI_Send(local_pixels, w * h * 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }

  delete[] localC;
  
  MPI_Finalize();
}