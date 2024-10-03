#ifndef SPUTTERER_TIMER_H
#define SPUTTERER_TIMER_H

// iterations to average over when smoothing times
constexpr int iter_avg = 25;
constexpr double time_const = 1.0 / iter_avg;

double exp_avg(double a, double b, double t) {
    return (1 - t) * a + t * b;
}

struct Timer {
    double physical_time = 0.0;
    double avg_time_compute = 0.0; 
    double avg_time_total = 0.0;
    double dt_smoothed = 0.0;
    void update_averages(double elapsed_compute, double elapsed_copy) {
        avg_time_compute = exp_avg(avg_time_compute, elapsed_compute, time_const);
        avg_time_total   = exp_avg(avg_time_total,   elapsed_copy,    time_const);
    }
};

#endif //SPUTTERER_TIMER_H
