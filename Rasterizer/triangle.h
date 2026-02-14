#pragma once

#include "mesh.h"
#include "colour.h"
#include "renderer.h"
#include "light.h"
#include <iostream>
#include <algorithm>
#include <cmath>

#include "optimizations.h"

#if USE_SIMD_OPTIMIZATION
#include <immintrin.h>  // SSE/AVX intrinsics
#endif

// Simple support class for a 2D vector
template <typename T = float>
class vec2D {
public:
    T x, y;

    // Default constructor initializes both components to 0
    vec2D() { x = y = T(0); };

    // Constructor initializes components with given values
    vec2D(T _x, T _y) : x(_x), y(_y) {}

    // Constructor initializes components from a vec4
    vec2D(vec4 v) {
        x = static_cast<T>(v[0]);
        y = static_cast<T>(v[1]);
    }

    // Display the vector components
    void display() { std::cout << x << '\t' << y << std::endl; }

    // Overloaded subtraction operator for vector subtraction
    vec2D operator- (vec2D& v) {
        vec2D q;
        q.x = x - v.x;
        q.y = y - v.y;
        return q;
    }
};

// Class representing a triangle for rendering purposes
class triangle {
    Vertex v[3];       // Vertices of the triangle
    float area;        // Area of the triangle
    colour col[3];     // Colors for each vertex of the triangle
    
    #if USE_STORE_VEC2D_INV_AREA_OPTIMIZATION
        float invArea;
        vec2D<float> zero, one, two;
        vec2D<float> edge01, edge12, edge20;
    #endif

public:
    // Constructor initializes the triangle with three vertices
    // Input Variables:
    // - v1, v2, v3: Vertices defining the triangle
    triangle(const Vertex& v1, const Vertex& v2, const Vertex& v3) {
        v[0] = v1;
        v[1] = v2;
        v[2] = v3;

        // Calculate the 2D area of the triangle
        vec2D e1 = vec2D(v[1].p - v[0].p);
        vec2D e2 = vec2D(v[2].p - v[0].p);
        area = std::fabs(e1.x * e2.y - e1.y * e2.x);
        
        #if USE_STORE_VEC2D_INV_AREA_OPTIMIZATION
            invArea = 1.0f / area;

            zero = vec2D(v[0].p);
            one = vec2D(v[1].p);
            two = vec2D(v[2].p);

            edge01.x = one.x - zero.x;
            edge01.y = one.y - zero.y;
            edge12.x = two.x - one.x;
            edge12.y = two.y - one.y;
            edge20.x = zero.x - two.x;
            edge20.y = zero.y - two.y;
        #endif
    }

    // Helper function to compute the cross product for barycentric coordinates
    // Input Variables:
    // - v1, v2: Edges defining the vector
    // - p: Point for which coordinates are being calculated
    float getC(vec2D<float> v1, vec2D<float> v2, vec2D<float> p) {
        vec2D e = v2 - v1;
        vec2D q = p - v1;
        return q.y * e.x - q.x * e.y;
    }

    // Compute barycentric coordinates for a given point
    // Input Variables:
    // - p: Point to check within the triangle
    // Output Variables:
    // - alpha, beta, gamma: Barycentric coordinates of the point
    // Returns true if the point is inside the triangle, false otherwise
    bool getCoordinates(vec2D<float> p, float& alpha, float& beta, float& gamma) {
        #if USE_STORE_VEC2D_INV_AREA_OPTIMIZATION
            alpha = getC(zero, one, p) * invArea;
            beta = getC(one, two, p) * invArea;
            gamma = getC(two, zero, p) * invArea;
        #else
            alpha = getC(vec2D(v[0].p), vec2D(v[1].p), p) / area;
            beta = getC(vec2D(v[1].p), vec2D(v[2].p), p) / area;
            gamma = getC(vec2D(v[2].p), vec2D(v[0].p), p) / area;
        #endif

        if (alpha < 0.f || beta < 0.f || gamma < 0.f) return false;
        return true;
    }

    // Template function to interpolate values using barycentric coordinates
    // Input Variables:
    // - alpha, beta, gamma: Barycentric coordinates
    // - a1, a2, a3: Values to interpolate
    // Returns the interpolated value
    template <typename T>
    T interpolate(float alpha, float beta, float gamma, T a1, T a2, T a3) {
        return (a1 * alpha) + (a2 * beta) + (a3 * gamma);
    }

    // Draw the triangle on the canvas
    // Input Variables:
    // - renderer: Renderer object for drawing
    // - L: Light object for shading calculations
    // - ka, kd: Ambient and diffuse lighting coefficients
    void draw(Renderer& renderer, Light& L, float ka, float kd) {
        vec2D minV, maxV;

        // Get the screen-space bounds of the triangle
        getBoundsWindow(renderer.canvas, minV, maxV);

        // Skip very small triangles
        if (area < 1.f) return;

        #if USE_LIGHT_NORM_OUT_OPTIMIZATION
            L.omega_i.normalise();
        #endif

        //std::cout << "Drawing triangle with x: " << maxV.x << "," << minV.x << " and y : " << maxV.y << "," << minV.y << std::endl;

        // Iterate over the bounding box and check each pixel
        for (int y = (int)(minV.y); y < (int)ceil(maxV.y); y++) {
            int x = (int)(minV.x);
            int xMax = (int)ceil(maxV.x);
            
            #if USE_SIMD_OPTIMIZATION && USE_STORE_VEC2D_INV_AREA_OPTIMIZATION
                // SIMD path: Process 4 pixels at a time
                __m128 py_vec = _mm_set1_ps((float)y);
                __m128 zero_x_vec = _mm_set1_ps(zero.x);
                __m128 zero_y_vec = _mm_set1_ps(zero.y);
                __m128 one_x_vec = _mm_set1_ps(one.x);
                __m128 one_y_vec = _mm_set1_ps(one.y);
                __m128 two_x_vec = _mm_set1_ps(two.x);
                __m128 two_y_vec = _mm_set1_ps(two.y);
                __m128 edge01_x_vec = _mm_set1_ps(edge01.x);
                __m128 edge01_y_vec = _mm_set1_ps(edge01.y);
                __m128 edge12_x_vec = _mm_set1_ps(edge12.x);
                __m128 edge12_y_vec = _mm_set1_ps(edge12.y);
                __m128 edge20_x_vec = _mm_set1_ps(edge20.x);
                __m128 edge20_y_vec = _mm_set1_ps(edge20.y);
                __m128 invArea_vec = _mm_set1_ps(invArea);
                __m128 zero_vec = _mm_setzero_ps();
                
                // Process 4 pixels at a time
                for (; x + 3 < xMax; x += 4) {
                    __m128 px_vec = _mm_set_ps((float)(x + 3), (float)(x + 2), (float)(x + 1), (float)x);
                    
                    // Compute qx0, qy0, qx1, qy1, qx2, qy2 for 4 pixels
                    __m128 qx0_vec = _mm_sub_ps(px_vec, zero_x_vec);
                    __m128 qy0_vec = _mm_sub_ps(py_vec, zero_y_vec);
                    __m128 qx1_vec = _mm_sub_ps(px_vec, one_x_vec);
                    __m128 qy1_vec = _mm_sub_ps(py_vec, one_y_vec);
                    __m128 qx2_vec = _mm_sub_ps(px_vec, two_x_vec);
                    __m128 qy2_vec = _mm_sub_ps(py_vec, two_y_vec);
                    
                    // Compute barycentric coordinates for 4 pixels - alpha first for early exit
                    __m128 alpha_vec = _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(qy0_vec, edge01_x_vec), _mm_mul_ps(qx0_vec, edge01_y_vec)), invArea_vec);
                    __m128 alpha_mask = _mm_cmpge_ps(alpha_vec, zero_vec);
                    
                    // Early exit if all alphas are negative
                    if (_mm_movemask_ps(alpha_mask) == 0) continue;
                    
                    // Compute beta
                    __m128 beta_vec = _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(qy1_vec, edge12_x_vec), _mm_mul_ps(qx1_vec, edge12_y_vec)), invArea_vec);
                    __m128 beta_mask = _mm_cmpge_ps(beta_vec, zero_vec);
                    
                    // Early exit if no pixels pass both alpha and beta
                    __m128 alpha_beta_mask = _mm_and_ps(alpha_mask, beta_mask);
                    if (_mm_movemask_ps(alpha_beta_mask) == 0) continue;
                    
                    // Compute gamma
                    __m128 gamma_vec = _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(qy2_vec, edge20_x_vec), _mm_mul_ps(qx2_vec, edge20_y_vec)), invArea_vec);
                    __m128 gamma_mask = _mm_cmpge_ps(gamma_vec, zero_vec);
                    __m128 inside_mask = _mm_and_ps(alpha_beta_mask, gamma_mask);
                    
                    // Extract mask and process pixels individually
                    int mask = _mm_movemask_ps(inside_mask);
                    
                    // Process each pixel that passed the test
                    float alpha_arr[4], beta_arr[4], gamma_arr[4];
                    _mm_store_ps(alpha_arr, alpha_vec);
                    _mm_store_ps(beta_arr, beta_vec);
                    _mm_store_ps(gamma_arr, gamma_vec);
                    
                    for (int i = 0; i < 4; i++) {
                        if (!(mask & (1 << i))) continue;
                        
                        float alpha = alpha_arr[i];
                        float beta = beta_arr[i];
                        float gamma = gamma_arr[i];
                        int px = x + i;

                        #if USE_EARLY_DEPTH_TEST_OPTIMIZATION
                            float depth = interpolate(beta, gamma, alpha, v[0].p[2], v[1].p[2], v[2].p[2]);
                            if (!(renderer.zbuffer(px, y) > depth && depth > 0.001f)) continue;
                        #endif

                        // Interpolate color, depth, and normals
                        colour c = interpolate(beta, gamma, alpha, v[0].rgb, v[1].rgb, v[2].rgb);
                        c.clampColour();

                        #if !USE_EARLY_DEPTH_TEST_OPTIMIZATION
                            float depth = interpolate(beta, gamma, alpha, v[0].p[2], v[1].p[2], v[2].p[2]);
                        #endif
                        
                        vec4 normal = interpolate(beta, gamma, alpha, v[0].normal, v[1].normal, v[2].normal);
                        #if !USE_AVOID_NORMAL_NORMALIZATION_OPTIMIZATION
                            normal.normalise();
                        #endif

                        // Perform Z-buffer test and apply shading
                        #if !USE_EARLY_DEPTH_TEST_OPTIMIZATION
                            if (!(renderer.zbuffer(px, y) > depth && depth > 0.001f)) continue;
                        #endif

                        // typical shader begin
                        #if !USE_LIGHT_NORM_OUT_OPTIMIZATION
                            L.omega_i.normalise();
                        #endif

                        float dot = std::max(vec4::dot(L.omega_i, normal), 0.0f);
                        colour a = (c * kd) * (L.L * dot) + (L.ambient * ka);
                        // typical shader end
                        unsigned char r, g, b;
                        a.toRGB(r, g, b);
                        renderer.canvas.draw(px, y, r, g, b);
                        renderer.zbuffer(px, y) = depth;
                    }
                }
            #endif
            
            // Scalar path: Process remaining pixels one at a time
            for (; x < xMax; x++) {
                #if USE_STORE_VEC2D_INV_AREA_OPTIMIZATION
                    float px = (float)x;
                    float py = (float)y;

                    // Compute alpha first for early exit
                    float qx0 = px - zero.x;
                    float qy0 = py - zero.y;
                    float alpha = (qy0 * edge01.x - qx0 * edge01.y) * invArea;
                    if (alpha < 0.f) continue;
                    
                    // Only compute beta if alpha passed
                    float qx1 = px - one.x;
                    float qy1 = py - one.y;
                    float beta = (qy1 * edge12.x - qx1 * edge12.y) * invArea;
                    if (beta < 0.f) continue;
                    
                    // Only compute gamma if alpha and beta passed
                    float qx2 = px - two.x;
                    float qy2 = py - two.y;
                    float gamma = (qy2 * edge20.x - qx2 * edge20.y) * invArea;
                    if (gamma < 0.f) continue;
                #else
                    // Check if the pixel lies inside the triangle
                    float alpha, beta, gamma;
                    if (!getCoordinates(vec2D((float)x, (float)y), alpha, beta, gamma)) continue;
                #endif

                #if USE_EARLY_DEPTH_TEST_OPTIMIZATION
                    float depth = interpolate(beta, gamma, alpha, v[0].p[2], v[1].p[2], v[2].p[2]);
                    if (!(renderer.zbuffer(x, y) > depth && depth > 0.001f)) continue;
                #endif

                // Interpolate color, depth, and normals
                colour c = interpolate(beta, gamma, alpha, v[0].rgb, v[1].rgb, v[2].rgb);
                c.clampColour();

                #if !USE_EARLY_DEPTH_TEST_OPTIMIZATION
                    float depth = interpolate(beta, gamma, alpha, v[0].p[2], v[1].p[2], v[2].p[2]);
                #endif
                
                vec4 normal = interpolate(beta, gamma, alpha, v[0].normal, v[1].normal, v[2].normal);
                #if !USE_AVOID_NORMAL_NORMALIZATION_OPTIMIZATION
                    normal.normalise();
                #endif
                

                // Perform Z-buffer test and apply shading
                #if !USE_EARLY_DEPTH_TEST_OPTIMIZATION
                    if (!(renderer.zbuffer(x, y) > depth && depth > 0.001f)) continue;
                #endif

                // typical shader begin
                #if !USE_LIGHT_NORM_OUT_OPTIMIZATION
                    L.omega_i.normalise();
                #endif

                float dot = std::max(vec4::dot(L.omega_i, normal), 0.0f);
                colour a = (c * kd) * (L.L * dot) + (L.ambient * ka); // using kd instead of ka for ambient
                // typical shader end
                unsigned char r, g, b;
                a.toRGB(r, g, b);
                renderer.canvas.draw(x, y, r, g, b);
                renderer.zbuffer(x, y) = depth;
            }
        }
    }

    // Draw the triangle on the canvas with Y-range constraints for multithreading
    // Input Variables:
    // - renderer: Renderer object for drawing
    // - L: Light object for shading calculations
    // - ka, kd: Ambient and diffuse lighting coefficients
    // - minV, maxV: Pre-calculated bounds of the triangle
    // - startY, endY: Y-coordinate range this thread is responsible for
    void draw(Renderer& renderer, Light& L, float ka, float kd, vec2D<int> minV, vec2D<int> maxV, int startY, int endY) {
        // Skip very small triangles
        if (area < 1.f) return;

        minV.y = std::max(minV.y, startY);
        maxV.y = std::min(maxV.y, endY);

        L.omega_i.normalise();

        // Iterate over the bounding box and check each pixel
        for (int y = minV.y; y < maxV.y; y++) {
            int x = minV.x;
            int xMax = maxV.x;
            
            #if USE_SIMD_OPTIMIZATION
                // SIMD path: Process 4 pixels at a time
                __m128 py_vec = _mm_set1_ps((float)y);
                __m128 zero_x_vec = _mm_set1_ps(zero.x);
                __m128 zero_y_vec = _mm_set1_ps(zero.y);
                __m128 one_x_vec = _mm_set1_ps(one.x);
                __m128 one_y_vec = _mm_set1_ps(one.y);
                __m128 two_x_vec = _mm_set1_ps(two.x);
                __m128 two_y_vec = _mm_set1_ps(two.y);
                __m128 edge01_x_vec = _mm_set1_ps(edge01.x);
                __m128 edge01_y_vec = _mm_set1_ps(edge01.y);
                __m128 edge12_x_vec = _mm_set1_ps(edge12.x);
                __m128 edge12_y_vec = _mm_set1_ps(edge12.y);
                __m128 edge20_x_vec = _mm_set1_ps(edge20.x);
                __m128 edge20_y_vec = _mm_set1_ps(edge20.y);
                __m128 invArea_vec = _mm_set1_ps(invArea);
                __m128 zero_vec = _mm_setzero_ps();
                
                // Process 4 pixels at a time
                for (; x + 3 < xMax; x += 4) {
                    __m128 px_vec = _mm_set_ps((float)(x + 3), (float)(x + 2), (float)(x + 1), (float)x);
                    
                    // Compute qx0, qy0, qx1, qy1, qx2, qy2 for 4 pixels
                    __m128 qx0_vec = _mm_sub_ps(px_vec, zero_x_vec);
                    __m128 qy0_vec = _mm_sub_ps(py_vec, zero_y_vec);
                    __m128 qx1_vec = _mm_sub_ps(px_vec, one_x_vec);
                    __m128 qy1_vec = _mm_sub_ps(py_vec, one_y_vec);
                    __m128 qx2_vec = _mm_sub_ps(px_vec, two_x_vec);
                    __m128 qy2_vec = _mm_sub_ps(py_vec, two_y_vec);
                    
                    // Compute barycentric coordinates for 4 pixels - alpha first for early exit
                    __m128 alpha_vec = _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(qy0_vec, edge01_x_vec), _mm_mul_ps(qx0_vec, edge01_y_vec)), invArea_vec);
                    __m128 alpha_mask = _mm_cmpge_ps(alpha_vec, zero_vec);
                    
                    // Early exit if all alphas are negative
                    if (_mm_movemask_ps(alpha_mask) == 0) continue;
                    
                    // Compute beta
                    __m128 beta_vec = _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(qy1_vec, edge12_x_vec), _mm_mul_ps(qx1_vec, edge12_y_vec)), invArea_vec);
                    __m128 beta_mask = _mm_cmpge_ps(beta_vec, zero_vec);
                    
                    // Early exit if no pixels pass both alpha and beta
                    __m128 alpha_beta_mask = _mm_and_ps(alpha_mask, beta_mask);
                    if (_mm_movemask_ps(alpha_beta_mask) == 0) continue;
                    
                    // Compute gamma
                    __m128 gamma_vec = _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(qy2_vec, edge20_x_vec), _mm_mul_ps(qx2_vec, edge20_y_vec)), invArea_vec);
                    __m128 gamma_mask = _mm_cmpge_ps(gamma_vec, zero_vec);
                    __m128 inside_mask = _mm_and_ps(alpha_beta_mask, gamma_mask);
                    
                    // Extract mask and process pixels individually
                    int mask = _mm_movemask_ps(inside_mask);
                    
                    // Process each pixel that passed the test
                    float alpha_arr[4], beta_arr[4], gamma_arr[4];
                    _mm_store_ps(alpha_arr, alpha_vec);
                    _mm_store_ps(beta_arr, beta_vec);
                    _mm_store_ps(gamma_arr, gamma_vec);

                    for (int i = 0; i < 4; i++) {
                        if (!(mask & (1 << i))) continue;
                        
                        float alpha = alpha_arr[i];
                        float beta = beta_arr[i];
                        float gamma = gamma_arr[i];
                        int px = x + i;

                        float depth = interpolate(beta, gamma, alpha, v[0].p[2], v[1].p[2], v[2].p[2]);
                        if (!(renderer.zbuffer(px, y) > depth && depth > 0.001f)) continue;

                        // Interpolate color, depth, and normals
                        colour c = interpolate(beta, gamma, alpha, v[0].rgb, v[1].rgb, v[2].rgb);
                        c.clampColour();
                        vec4 normal = interpolate(beta, gamma, alpha, v[0].normal, v[1].normal, v[2].normal);

                        float dot = std::max(vec4::dot(L.omega_i, normal), 0.0f);
                        colour a = (c * kd) * (L.L * dot) + (L.ambient * ka);

                        unsigned char r, g, b;
                        a.toRGB(r, g, b);
                        renderer.canvas.draw(px, y, r, g, b);
                        renderer.zbuffer(px, y) = depth;
                    }
                }
            #endif
            
            // Scalar path: Process remaining pixels one at a time
            for (; x < xMax; x++) {
                float px = (float)x;
                float py = (float)y;

                float qx0 = px - zero.x;
                float qy0 = py - zero.y;
                float alpha = (qy0 * edge01.x - qx0 * edge01.y) * invArea;
                if (alpha < 0.f) continue;
                
                float qx1 = px - one.x;
                float qy1 = py - one.y;
                float beta = (qy1 * edge12.x - qx1 * edge12.y) * invArea;
                if (beta < 0.f) continue;

                float qx2 = px - two.x;
                float qy2 = py - two.y;
                float gamma = (qy2 * edge20.x - qx2 * edge20.y) * invArea;
                if (gamma < 0.f) continue;

                float depth = interpolate(beta, gamma, alpha, v[0].p[2], v[1].p[2], v[2].p[2]);
                if (!(renderer.zbuffer(x, y) > depth && depth > 0.001f)) continue;
                

                // Interpolate color, depth, and normals
                colour c = interpolate(beta, gamma, alpha, v[0].rgb, v[1].rgb, v[2].rgb);
                c.clampColour();
                vec4 normal = interpolate(beta, gamma, alpha, v[0].normal, v[1].normal, v[2].normal);

                float dot = std::max(vec4::dot(L.omega_i, normal), 0.0f);
                colour a = (c * kd) * (L.L * dot) + (L.ambient * ka);

                unsigned char r, g, b;
                a.toRGB(r, g, b);
                renderer.canvas.draw(x, y, r, g, b);
                renderer.zbuffer(x, y) = depth;
            }
        }
    }

    // Compute the 2D bounds of the triangle
    // Output Variables:
    // - minV, maxV: Minimum and maximum bounds in 2D space
    template <typename T = float>
    void getBounds(vec2D<T>& minV, vec2D<T>& maxV) {
        minV = vec2D<T>(v[0].p);
        maxV = vec2D<T>(v[0].p);
        for (unsigned int i = 1; i < 3; i++) {
            minV.x = std::min(minV.x, static_cast<T>(v[i].p[0]));
            minV.y = std::min(minV.y, static_cast<T>(v[i].p[1]));
            maxV.x = std::max(maxV.x, static_cast<T>(v[i].p[0]));
            maxV.y = std::max(maxV.y, static_cast<T>(v[i].p[1]));
        }
    }

    // Compute the 2D bounds of the triangle, clipped to the canvas
    // Input Variables:
    // - canvas: Reference to the rendering canvas
    // Output Variables:
    // - minV, maxV: Clipped minimum and maximum bounds
    void getBoundsWindow(GamesEngineeringBase::Window& canvas, vec2D<float> &minV, vec2D<float> &maxV) {
        getBounds(minV, maxV);
        minV.x = std::max(minV.x, static_cast<float>(0));
        minV.y = std::max(minV.y, static_cast<float>(0));
        maxV.x = std::min(maxV.x, static_cast<float>(canvas.getWidth()));
        maxV.y = std::min(maxV.y, static_cast<float>(canvas.getHeight()));
    }

    void getBoundsWindow(GamesEngineeringBase::Window& canvas, vec2D<int> &minV, vec2D<int> &maxV) {
        getBounds(minV, maxV);
        minV.x = std::max(minV.x, static_cast<int>(0));
        minV.y = std::max(minV.y, static_cast<int>(0));
        maxV.x = std::min(maxV.x, static_cast<int>(canvas.getWidth()));
        maxV.y = std::min(maxV.y, static_cast<int>(canvas.getHeight()));
    }

    // Debugging utility to display the triangle bounds on the canvas
    // Input Variables:
    // - canvas: Reference to the rendering canvas
    void drawBounds(GamesEngineeringBase::Window& canvas) {
        vec2D minV, maxV;
        getBounds(minV, maxV);

        for (int y = (int)minV.y; y < (int)maxV.y; y++) {
            for (int x = (int)minV.x; x < (int)maxV.x; x++) {
                canvas.draw(x, y, 255, 0, 0);
            }
        }
    }

    // Debugging utility to display the coordinates of the triangle vertices
    void display() {
        for (unsigned int i = 0; i < 3; i++) {
            v[i].p.display();
        }
        std::cout << std::endl;
    }
};
