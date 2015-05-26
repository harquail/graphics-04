// Hi Nook and Kiko!

#include "scene.h"
#include "intersect.h"
#include "montecarlo.h"
#include "animation.h"

#include <thread>
using std::thread;

// modify the following line to disable/enable parallel execution of the pathtracer
bool parallel_pathtrace = true;

image3f pathtrace(Scene* scene, bool multithread);
void pathtrace(Scene* scene, image3f* image, RngImage* rngs, int offset_row, int skip_row, bool verbose);


// lookup texture value
vec3f lookup_scaled_texture(vec3f value, image3f* texture, vec2f uv, bool tile = false) {
    if(not texture) return value;
    
    bool filtering = false;
    
    // texture tiling
    if(tile){
        auto i = (int)(uv.x*(texture->width()-1));
        auto j = (int)(uv.y*(texture->height()-1));
        
        i = i % texture->width();
        if (i < 0){
            i = i + texture->width();
        }
        
        j = j % texture->width();
        if (j < 0){
            j = j + texture->width();
        }
        
//        auto i = (int)(uv.x*(texture->width()-1));
//        auto j = (int)(uv.y*(texture->height()-1));
//
        if(filtering){
        auto s = uv.x * (texture->width()-1) - i;
        auto t = uv.y * (texture->height()-1) - j;
//
        auto i2 = i + 1;
        auto j2 = j + 1;
//
        auto c1 = texture->at(i,j)*(1-s)*(1-t);
        auto c2 = texture->at(i, j2)*(1-s)*t;
        auto c3 = texture->at(i2,j)*s*(1-t);
        auto c4 = texture->at(i2,j2)*s*t;
        auto c = c1 + c2 + c3 + c4;
//
            return value * c;
        }
        else{
            return value * texture->at(i,j);
        }
        
    }
    else{
        // for now, simply clamp texture coords
        auto u = clamp(uv.x, 0.0f, 1.0f);
        auto v = clamp(uv.y, 0.0f, 1.0f);
        return value * texture->at(u*(texture->width()-1), v*(texture->height()-1));
    }
}

// compute the brdf
vec3f eval_brdf(vec3f kd, vec3f ks, float n, vec3f v, vec3f l, vec3f norm, bool microfacet) {
    if (not microfacet) {
            auto h = normalize(v+l);
        return kd/pif + ks*(n+8)/(8*pif) * pow(max(0.0f,dot(norm,h)),n);
    } else {
        // help from this: http://simonstechblog.blogspot.com/2011/12/microfacet-brdf.html
            auto h = normalize( v + l);
            auto dist = (n + 2.0f) / (2.0f*pif) * pow(max(0.0f, dot(norm,h)),n);
            auto fresnel = ks + (one3f - ks)*pow(1.0f - dot(h,l), 5.0f);
            float geo = min(1.0f,2.0f * dot(h,norm) * dot(v,norm) / dot(v,h));
            geo = min(geo,(2.0f * dot(h,norm)*dot(l,norm) / dot(l,h)));
        return (dist * geo * fresnel)/(4.0f * dot(l,norm) * dot(v,norm));
    }
}

// evaluate the environment map
vec3f eval_env(vec3f ke, image3f* ke_txt, vec3f dir) {
    vec2f uv;
    uv.x = atan2f(dir.x,dir.z)/(2.0f*pif);
    uv.y = 1.0f-acosf(dir.y)/pif;
    auto vector = lookup_scaled_texture(ke, ke_txt, uv);
    return vector;
}

// compute the color corresponing to a ray by pathtrace
vec3f pathtrace_ray(Scene* scene, ray3f ray, Rng* rng, int depth) {
    // get scene intersection
    auto intersection = intersect(scene,ray);
    //printf("%d\n", depth);
    
    // if not hit, return background (looking up the texture by converting the ray direction to latlong around y)
    if(not intersection.hit) {
        return eval_env(scene->background, scene->background_txt, ray.d);
    }
    
    auto pos = intersection.pos;
    auto norm = intersection.norm;
    auto v = -ray.d;
    
    // compute material values by looking up textures
    auto ke = lookup_scaled_texture(intersection.mat->ke, intersection.mat->ke_txt, intersection.texcoord, true);
    auto kd = lookup_scaled_texture(intersection.mat->kd, intersection.mat->kd_txt, intersection.texcoord, true);
    auto ks = lookup_scaled_texture(intersection.mat->ks, intersection.mat->ks_txt, intersection.texcoord, true);
    auto n = intersection.mat->n;
    auto mf = intersection.mat->microfacet;
    
    // accumulate color starting with ambient
    auto c = scene->ambient * kd;
    
    // add emission if on the first bounce
    if(depth == 0 and dot(v,norm) > 0) c += ke;
    
    // foreach point light
    for(auto light : scene->lights) {
        // compute light response
        auto cl = light->intensity / (lengthSqr(light->frame.o - pos));
        // compute light direction
        auto l = normalize(light->frame.o - pos);
        // compute the material response (brdf*cos)
        auto brdfcos = max(dot(norm,l),0.0f) * eval_brdf(kd, ks, n, v, l, norm, mf);
        // multiply brdf and light
        auto shade = cl * brdfcos;
        // check for shadows and accumulate if needed
        if(shade == zero3f) continue;
        // if shadows are enabled
        if(scene->path_shadows) {
            // perform a shadow check and accumulate
            if(not intersect_shadow(scene,ray3f::make_segment(pos,light->frame.o))) c += shade;
        } else {
            // else just accumulate
            c += shade;
        }
    }
    
    // foreach surface
    for (auto surface : scene->surfaces) {
        // skip if no emission from surface
        if (surface->mat->ke == zero3f)
            continue;
        
        vec3f lnorm;
        float area;
        auto lpos = zero3f;
        vec2f texcoords;
        // todo: pick a point on the surface, grabbing normal, area, and texcoord
        // check if quad
        if (surface->isquad == true) {
            // generate a 2d random number
            auto num = rng->next_vec2f();
            
            // compute light position, normal, area
            lpos.x = (num.x - .5)*2*surface->radius;
            lpos.y = (num.y - .5)*2*surface->radius;
            lpos = transform_point_from_local(surface->frame, lpos);
            
            lnorm = transform_normal_from_local(surface->frame, z3f);
            area = 4*pow(surface->radius,2);
            // set tex coords as random value got before
            texcoords = num;
        }
        // else if sphere
        else {
            // generate a 2d random number
            auto num = rng->next_vec2f();
            
            // compute light position, normal, area
            lpos = sample_direction_spherical_uniform(num);
            lnorm = normalize(lpos);
            area = 4*PI*pow(surface->radius, 2);
            // set tex coords as random value got before
            texcoords = num;
            
        }
        
        // get light emission from material and texture
        auto le = lookup_scaled_texture(surface->mat->ke, surface->mat->ke_txt, texcoords);
        
        // compute light direction from pos to lpos
        auto l = normalize(lpos - pos);
        
        // compute light response (ke * area * cos_of_light / dist^2)
        auto cl = le * area * dot(lnorm,-l) / distSqr(pos,lpos);
        
        // compute the material response (brdf*cos)
        auto brdfcos = max(dot(norm,l),0.0f) * eval_brdf(kd, ks, n, v, l, norm, mf);
        
        // multiply brdf and light
        auto shade = cl * brdfcos;
        
        // check for shadows and accumulate if needed
        if(shade == zero3f) continue;
        // if shadows are enabled
        if(scene->path_shadows) {
            // perform a shadow check and accumulate
            if(not intersect_shadow(scene,ray3f::make_segment(pos,lpos))) c += shade;
        } else {
            // else just accumulate
            c += shade;
        }
    }
    
    
    // todo: sample the brdf for environment illumination if the environment is there
    // if scene->background is not zero3f
    if (scene->background != zero3f) {
        
        // pick direction and pdf
        auto brdf = sample_brdf(kd, ks, n, v, norm, rng->next_vec2f(), rng->next_float());
        auto text = eval_env(scene->background, scene->background_txt, brdf.first);
        
        // compute the material response (brdf*cos)
        auto brdfcos = max(dot(norm,brdf.first),0.0f) * eval_brdf(kd, ks, n, v, brdf.first, norm, mf);
        
        
        // todo: accumulate response scaled by brdf*cos/pdf
        auto shade = text * brdfcos / brdf.second;
        
        // if material response not zero3f
        if (brdfcos != zero3f) {
            
            // if shadows are enabled
            if (scene->path_shadows) {
                // perform a shadow check and accumulate
                if(not intersect_shadow(scene,ray3f::make_segment(pos, brdf.first))){
                    c += shade;
                }
            }
            // else just accumulate
            else {
                
                c += shade;
            }
        }
    }

    
    // todo: sample the brdf for indirect illumination
    // if kd and ks are not zero3f and haven't reach max_depth
    if ( (kd != zero3f || ks != zero3f) && depth < scene->path_max_depth) {
        
        // printf("I am here!");
        // pick direction and pdf
        auto brdf = sample_brdf(kd, ks, n, v, norm, rng->next_vec2f(), rng->next_float());
        // auto text = eval_env(scene->background, scene->background_txt, brdf.first);
        
        ray3f ray2 = ray3f(pos,brdf.first);
        
        
        // compute the material response (brdf*cos)
        auto brdfcos = max(dot(norm,brdf.first),0.0f) * eval_brdf(kd, ks, n, v, brdf.first, norm, mf);
        
        auto theta = .3; // russian roulette
        auto rand = rng->next_float(); // russian roulette
        if (rand <= theta) { // russian roulette
        c += brdfcos * pathtrace_ray(scene, ray, rng, depth+1)/ brdf.second;
        c /= (1-theta); // russian roulette
        } // russian roulette


        // accumulate recersively scaled by brdf*cos/pdf
        c += brdfcos * pathtrace_ray(scene, ray2, rng, depth+1)/ brdf.second;
        //c += shade;
        
    }
    
    // if the material has reflections
    if(not (intersection.mat->kr == zero3f)) {
        
        if (intersection.mat->blurry_reflection == 0.0f){
            // create the reflection ray
            auto rr = ray3f(intersection.pos,reflect(ray.d,intersection.norm));
            // accumulate the reflected light (recursive call) scaled by the material reflection
            c += intersection.mat->kr * pathtrace_ray(scene,rr,rng,depth+1);
        }
        
        else {
            
            auto randomR = rng->next_vec3f()*intersection.mat->blurry_reflection;
            
            auto direction = normalize(reflect(ray.d+randomR,intersection.norm));
            
            auto rr = ray3f(intersection.pos,direction);
            
            c += intersection.mat->kr * pathtrace_ray(scene,rr,rng,depth+1);

        }

    }
    
    // return the accumulated color
    return c;
}


// runs the raytrace over all tests and saves the corresponding images
int main(int argc, char** argv) {
    auto args = parse_cmdline(argc, argv,
        { "04_pathtrace", "raytrace a scene",
            {  {"resolution",     "r", "image resolution", typeid(int),    true,  jsonvalue() } },
            {  {"scene_filename", "",  "scene filename",   typeid(string), false, jsonvalue("scene.json") },
               {"image_filename", "",  "image filename",   typeid(string), true,  jsonvalue("") } }
        });
    
    auto scene_filename = args.object_element("scene_filename").as_string();
    Scene* scene = nullptr;
    if(scene_filename.length() > 9 and scene_filename.substr(0,9) == "testscene") {
        int scene_type = atoi(scene_filename.substr(9).c_str());
        scene = create_test_scene(scene_type);
        scene_filename = scene_filename + ".json";
    } else {
        scene = load_json_scene(scene_filename);
    }
    error_if_not(scene, "scene is nullptr");
    
    auto image_filename = (args.object_element("image_filename").as_string() != "") ?
        args.object_element("image_filename").as_string() :
        scene_filename.substr(0,scene_filename.size()-5)+".png";
    
    if(not args.object_element("resolution").is_null()) {
        scene->image_height = args.object_element("resolution").as_int();
        scene->image_width = scene->camera->width * scene->image_height / scene->camera->height;
    }
    
    // NOTE: acceleration structure does not support animations
    message("reseting animation...\n");
    animate_reset(scene);
    
    message("accelerating...\n");
    accelerate(scene);
    
    message("rendering %s...\n", scene_filename.c_str());
    auto image = pathtrace(scene, parallel_pathtrace);
    
    message("saving %s...\n", image_filename.c_str());
    write_png(image_filename, image, true);
    
    delete scene;
    message("done\n");
}


/////////////////////////////////////////////////////////////////////
// Rendering Code


// pathtrace an image
void pathtrace(Scene* scene, image3f* image, RngImage* rngs, int offset_row, int skip_row, bool verbose) {
    if(verbose) message("\n  rendering started        ");
    // foreach pixel
    for(auto j = offset_row; j < scene->image_height; j += skip_row ) {
        if(verbose) message("\r  rendering %03d/%03d        ", j, scene->image_height);
        for(auto i = 0; i < scene->image_width; i ++) {
            // init accumulated color
            image->at(i,j) = zero3f;
            // grab proper random number generator
            auto rng = &rngs->at(i, j);
            // foreach sample
            for(auto jj : range(scene->image_samples)) {
                for(auto ii : range(scene->image_samples)) {
                    // compute ray-camera parameters (u,v) for the pixel and the sample
                    auto u = (i + (ii + rng->next_float())/scene->image_samples) /
                        scene->image_width;
                    auto v = (j + (jj + rng->next_float())/scene->image_samples) /
                        scene->image_height;
                    // compute camera ray
                    auto ray = transform_ray(scene->camera->frame,
                        ray3f(zero3f,normalize(vec3f((u-0.5f)*scene->camera->width,
                                                     (v-0.5f)*scene->camera->height,-1))));
                    // set pixel to the color raytraced with the ray
                    image->at(i,j) += pathtrace_ray(scene,ray,rng,0);
                }
            }
            // scale by the number of samples
            image->at(i,j) /= (scene->image_samples*scene->image_samples);
        }
    }
    if(verbose) message("\r  rendering done        \n");
    
}

// pathtrace an image with multithreading if necessary
image3f pathtrace(Scene* scene, bool multithread) {
    // allocate an image of the proper size
    auto image = image3f(scene->image_width, scene->image_height);
    
    // create a random number generator for each pixel
    auto rngs = RngImage(scene->image_width, scene->image_height);

    // if multitreaded
    if(multithread) {
        // get pointers
        auto image_ptr = &image;
        auto rngs_ptr = &rngs;
        // allocate threads and pathtrace in blocks
        auto threads = vector<thread>();
        auto nthreads = thread::hardware_concurrency();
        for(auto tid : range(nthreads)) threads.push_back(thread([=](){
            return pathtrace(scene,image_ptr,rngs_ptr,tid,nthreads,tid==0);}));
        for(auto& thread : threads) thread.join();
    } else {
        // pathtrace all rows
        pathtrace(scene, &image, &rngs, 0, 1, true);
    }
    
    // done
    return image;
}


