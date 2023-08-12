__constant float EPSILON = 0.00003f;
__constant float PI = 3.14159265359f;
__constant int MAX_DEPTH = 10;

typedef struct Ray{
	float3 origin;
	float3 dir;
} Ray;

typedef enum { DIFF, SPEC, REFR } Refl_t;

typedef struct Sphere{
	float radius;
	float3 position;
	float3 color;
	float3 emission;
	Refl_t refl;
} Sphere;

static float get_random(unsigned int *seed0, unsigned int *seed1) {

	*seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);  
	*seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

	unsigned int ires = ((*seed0) << 16) + (*seed1);

	union {
		float f;
		unsigned int ui;
	} res;

	res.ui = (ires & 0x007fffff) | 0x40000000;
	return (res.f - 2.0f) / 2.0f;
}

Ray createCamera(const int x_coord, const int y_coord, const int width, const int height){

	float fx = (float)x_coord / (float)width;
	float fy = (float)y_coord / (float)height;

	float aspect_ratio = (float)(width) / (float)(height);
	float fx2 = (fx - 0.5f) * aspect_ratio;
	float fy2 = fy - 0.5f;

	float3 pixel_pos = (float3)(fx2, -fy2, 0.0f);

	Ray ray;
	ray.origin = (float3)(0.0f, 0.1f, 1.5f);
	ray.dir = normalize(pixel_pos - ray.origin);

	return ray;
}

float intersect_sphere(__constant Sphere* sphere, const Ray* ray)
{
    float3 rayToCenter = sphere->position - ray->origin;
    float b = dot(rayToCenter, ray->dir);
    float c = dot(rayToCenter, rayToCenter) - sphere->radius*sphere->radius;
    float disc = b * b - c;

    if (disc < 0.0f) return 0.0f;
    else disc = sqrt(disc);

    float t = b - disc;
    if (t > EPSILON) return t;

    t = b + disc;
    if (t > EPSILON) return t;

    return 0.0f;
}



float intersect(const Ray r, int* id, const int num_spheres, __constant Sphere* spheres) {
    float t = INFINITY;
    for (int i = 0; i < num_spheres; i++) {
        float d = intersect_sphere(&spheres[i],&r);
        if (d > 0 && d < t) {
            t = d;
            *id = i;
        }
    }
    return t;
}


float3 radiance(__constant Sphere* spheres, const Ray* camray, const int sphere_count, unsigned int* seed0, unsigned int* seed1){
    Ray r = *camray;
    float3 canvas = (float3)(0.0f, 0.0f, 0.0f);
    float3 mask = (float3)(1.0f, 1.0f, 1.0f);
    for (int depth = 0; depth < MAX_DEPTH; depth++) {
        int id = 0;
        float t = intersect(r, &id, sphere_count, spheres);
        if (t == INFINITY) {
            return canvas;
        }
        const Sphere obj = spheres[id];
        float3 x = r.origin + r.dir * t;
        float3 n = normalize(x - obj.position);
        float3 nl = dot(n, r.dir) < 0 ? n : n * -1;
        float3 f = obj.color;
        canvas += mask * obj.emission;
        mask *= f;
        if (obj.refl == DIFF) {
            float r1 = 2 * PI * get_random(seed0,seed1),
                   r2 = get_random(seed0,seed1),
                   r2s = sqrt(r2);
            float3 w = nl,
                   u = normalize(cross(fabs(w.x) > 0.1 ? (float3)(0, 1, 0) : (float3)(1, 0, 0), w)),
                   v = cross(w,u);
            float3 d = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2));
            r.origin = x + nl * EPSILON;
            r.dir = d;
        } else if (obj.refl == SPEC) {
            float3 refl_dir = r.dir - n * 2 * dot(n,r.dir);
            float3 d = normalize(refl_dir);
            r.origin = x + nl * EPSILON;
            r.dir = d;
        } else if (obj.refl == REFR) {
            bool into = dot(n,nl) > 0;
            float nc = 1,
                  nt = 1.5,
                  nnt = into ? nc/nt : nt/nc,
                  ddn = dot(r.dir,nl),
                  cos2t = 1 - nnt*nnt*(1 - ddn*ddn);

            if(cos2t < 0){
                float3 refl_dir = r.dir - n * 2 * dot(n,r.dir);
                float3 d = normalize(refl_dir);
                r.origin = x + nl * EPSILON;
                r.dir = d;
            } else {
                float3 tdir;
                if(into){
                    tdir=normalize(r.dir*nnt-n*(ddn*nnt+sqrt(cos2t)));
                } else {
                    tdir=normalize(r.dir*nnt+n*(ddn*nnt+sqrt(cos2t)));
                }
                r.origin=x-nl*EPSILON;
                r.dir=tdir;
            }
        }
    }
    return canvas;
}

__kernel void render(__constant Sphere* spheres, const int width, const int height, const int sphere_count, __global float3* final_canvas,const int samps)
{
	unsigned int work_item_id = get_global_id(0);	
	unsigned int x_coord = work_item_id % width;		
	unsigned int y_coord = work_item_id / width;			
	
	unsigned int seed0 = x_coord;
	unsigned int seed1 = y_coord;

	Ray camray = createCamera(x_coord, y_coord, width, height);

	float3 finalcolor = (float3)(0.0f, 0.0f, 0.0f);
	float invsamps = 1.0f / samps;

	for (int i = 0; i < samps; i++)
		finalcolor += radiance(spheres, &camray, sphere_count, &seed0, &seed1) * invsamps;

	final_canvas[work_item_id] = finalcolor;
}