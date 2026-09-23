#pragma once
#include "camera.h"
#include "../core/vec3.h"
#include <algorithm>
#include <cmath>

// 6-DOF Interactive fly camera for real-time viewport navigation.
// Supports pitch/yaw orientation, movement along view-relative forward/right,
// elevation along world-up, sprint/slow modifiers, and dynamic FOV zoom.
class fly_camera {
public:
    vec3 eye{0.0, 1.55, 3.8};
    double yaw = -90.0;     // degrees, -90 points along -Z
    double pitch = -14.0;   // degrees, negative looks slightly down
    double vfov = 37.0;     // vertical FOV in degrees
    double speed = 3.0;     // base movement speed in units/sec
    double mouse_sens = 0.12; // degrees per mouse delta pixel

    fly_camera() = default;

    void init_from_lookat(const vec3 &lookfrom, const vec3 &lookat, double vfov_deg) {
        eye = lookfrom;
        vfov = vfov_deg;
        vec3 d = unit_vector(lookat - lookfrom);
        pitch = std::asin(std::clamp(d.y(), -0.9999, 0.9999)) * (180.0 / 3.1415926535897932385);
        yaw = std::atan2(d.z(), d.x()) * (180.0 / 3.1415926535897932385);
    }

    void init_from_camera(const camera &c, double vfov_deg) {
        eye = c.eye();
        vfov = vfov_deg;
        vec3 d = c.dir();
        pitch = std::asin(std::clamp(d.y(), -0.9999, 0.9999)) * (180.0 / 3.1415926535897932385);
        yaw = std::atan2(d.z(), d.x()) * (180.0 / 3.1415926535897932385);
    }

    vec3 forward_dir() const {
        double ry = yaw * (3.1415926535897932385 / 180.0);
        double rp = pitch * (3.1415926535897932385 / 180.0);
        return unit_vector(vec3(std::cos(rp) * std::cos(ry), std::sin(rp), std::cos(rp) * std::sin(ry)));
    }

    vec3 right_dir() const {
        return unit_vector(cross(forward_dir(), vec3(0, 1, 0)));
    }

    camera build_camera(double aspect, double aperture = 0.0, double focus_dist = 1.0) const {
        vec3 fwd = forward_dir();
        vec3 target = eye + fwd * focus_dist;
        return camera(eye, target, vec3(0, 1, 0), vfov, aspect, aperture, focus_dist);
    }

    bool process_keyboard(double dt, bool forward, bool backward, bool left, bool right,
                          bool up, bool down, bool sprint, bool slow) {
        double cur_speed = speed;
        if (sprint) cur_speed *= 3.0;
        if (slow) cur_speed *= 0.25;
        double dist = cur_speed * dt;

        vec3 fwd = forward_dir();
        vec3 rgt = right_dir();
        vec3 world_up(0, 1, 0);

        vec3 move(0, 0, 0);
        if (forward)  move += fwd;
        if (backward) move -= fwd;
        if (right)    move += rgt;
        if (left)     move -= rgt;
        if (up)       move += world_up;
        if (down)     move -= world_up;

        if (move.length_squared() > 1e-8) {
            eye += unit_vector(move) * dist;
            return true;
        }
        return false;
    }

    bool process_mouse(double dx, double dy) {
        if (std::abs(dx) < 1e-4 && std::abs(dy) < 1e-4)
            return false;
        yaw += dx * mouse_sens;
        pitch -= dy * mouse_sens; // screen Y down is positive
        if (pitch > 89.0) pitch = 89.0;
        if (pitch < -89.0) pitch = -89.0;
        return true;
    }

    bool process_scroll(double dy) {
        if (std::abs(dy) < 1e-4) return false;
        vfov -= dy * 2.0;
        if (vfov < 5.0) vfov = 5.0;
        if (vfov > 140.0) vfov = 140.0;
        return true;
    }
};
