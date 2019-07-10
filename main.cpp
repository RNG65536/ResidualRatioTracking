#include <iostream>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <random>
#include <algorithm>
#include <vector>
#include <memory>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <iomanip>
using glm::vec2;
using std::cout;
using std::endl;
std::uniform_real_distribution<double> dist(0.0, 1.0);
std::mt19937_64 rng;
double randf()
{
    return dist(rng);
}

std::string title;
void set_window_title(double mse)
{
    std::stringstream ss;
    ss << std::setiosflags(std::ios::fixed) << std::setprecision(6);
    ss << "RMSE : " << std::sqrt(mse);
    title = ss.str();
    glutSetWindowTitle(title.c_str());
}

int width = 300;
int height = 300;

//////////////////////////////////////////////////////////////////////////
static double max_dist = 0;
static double max_cost = 0;

double sample_free_path(double sigma, double rnd)
{
    double s = -std::log(rnd + DBL_MIN) / sigma; // log(0) -> -INF
    if (s > max_dist)
    {
        max_dist = s;
    }
    return s;
}

bool is_null_collision(double sigma, double majorant, double rnd)
{
    return rnd > sigma / majorant;
}

double null_ratio(double sigma, double majorant)
{
    return 1.0 - sigma / majorant;
}

// one for each pixel
class TransmittanceEstimator
{
private:
    double m_sigma;
    double m_analytic_transmittance;

protected:
    const double m_length = 2.0;
    double m_majorant; // must be higher than sigma
    double m_sigma_control;

    double m_estimate_sum = 0;
    uint64_t m_cost_sum = 0;
    uint64_t m_num_samples = 0;

public:
    TransmittanceEstimator(double sampling_efficiency, double sigma)
        : m_sigma(sigma)
    {
        // density distribution :

        // 1) constant
        m_analytic_transmittance = std::exp(-m_sigma * m_length);
        m_majorant = sigma / sampling_efficiency;
        m_sigma_control = sigma; // min, max, average extinction coefficients are all the same

        // 2) linear
//         m_analytic_transmittance = std::exp(-m_sigma * m_length * 3.0 / 2.0);
//         m_majorant = (2 * sigma) / sampling_efficiency;
//         // m_sigma_control = sigma; // min extinction coefficient
//         // m_sigma_control = 2 * sigma; // max extinction coefficient
//         m_sigma_control = 1.5 * sigma; // average extinction coefficient (mostly optimal)

        // 3) parabolic
//         m_analytic_transmittance = std::exp(-m_sigma * m_length / 3.0);
//         m_majorant = sigma / sampling_efficiency;
//         // m_sigma_control = 0; // min extinction coefficient
//         // m_sigma_control = sigma; // max extinction coefficient
//         m_sigma_control = sigma / 3.0; // average extinction coefficient (mostly optimal)
    }

    float sigma(float p) const
    {
        // density distribution :

        // 1) constant
        return m_sigma;

        // 2) linear
//         return m_sigma * (1 + p / m_length);

        // 3) parabolic
//         return m_sigma * (p / m_length) * (p / m_length);
    }

    virtual void sample() = 0;

    double estimate_transmittance() const
    {
        if (m_num_samples == 0)
        {
            return 0;
        }
        return m_estimate_sum / m_num_samples;
    }
    double reference_transmittance() const
    {
        return m_analytic_transmittance;
    }
    double mean_cost() const
    {
        double m = double(m_cost_sum) / m_num_samples;
        return m;
    }
    double square_error() const
    {
        float d = (estimate_transmittance() - reference_transmittance());
        return d * d;
    }
};

class DeltaTrackingEstimator : public TransmittanceEstimator
{
private:
    void sample()
    {
        // delta tracking
        double e = 1; // transmittance estimate
        double p = 0;
        for (;;)
        {
            m_cost_sum++;

            p += sample_free_path(this->m_majorant, randf());

            if (p >= m_length)
            {
                break;
            }

            if (!is_null_collision(sigma(p), m_majorant, randf()))
            {
                e = 0;
                break;
            }
        }

        m_num_samples++;
        m_estimate_sum += e;
    }

public:
    DeltaTrackingEstimator(double sampling_efficiency, double sigma)
        : TransmittanceEstimator(sampling_efficiency, sigma)
    {

    }
};

class RatioTrackingEstimator : public TransmittanceEstimator
{
private:
    void sample()
    {
        // ratio tracking
        double e = 1; // transmittance estimate
        double p = 0;
        for (;;)
        {
            m_cost_sum++;

            p += sample_free_path(m_majorant, randf());

            if (p >= m_length)
            {
                break;
            }

            e *= null_ratio(sigma(p), m_majorant);
        }

        m_num_samples++;
        m_estimate_sum += e;
    }

public:
    RatioTrackingEstimator(double sampling_efficiency, double sigma)
        : TransmittanceEstimator(sampling_efficiency, sigma)
    {

    }
};

class ResidualRatioTrackingEstimator : public TransmittanceEstimator
{
private:
    void sample()
    {
        // ratio tracking
        double e = 1; // transmittance estimate
        double e_c = std::exp(-m_sigma_control * m_length);
        double p = 0;
        for (;;)
        {
            m_cost_sum++;

            p += sample_free_path(m_majorant, randf());

            if (p >= m_length)
            {
                break;
            }

            e *= null_ratio(sigma(p) - m_sigma_control, m_majorant);
        }
        e *= e_c;

        m_num_samples++;
        m_estimate_sum += e;
    }

public:
    ResidualRatioTrackingEstimator(double sampling_efficiency, double sigma)
        : TransmittanceEstimator(sampling_efficiency, sigma)
    {

    }
};

typedef enum { DrawEstimate, DrawReference, DrawCost } DrawType;

class VolumeSampler
{
    std::vector<std::unique_ptr<TransmittanceEstimator>> ss;
    DrawType type = DrawEstimate;
public:

    VolumeSampler()
    {
        reset<DeltaTrackingEstimator>();
    }

    void info()
    {
        cout << "max dist = " << max_dist << endl;
        cout << "max cost = " << max_cost << endl;
        cout << endl;
    }

    void toggle(DrawType t)
    {
        type = t;
    }

    // configuration
    template<typename EstimatorType>
    void reset()
    {
        max_dist = 0;
        max_cost = 0;

        ss.resize(width * height);

        {
            for (int j = 0; j < height; j++)
            {
                for (int i = 0; i < width; i++)
                {
                    double x = (i + 0.5) / width;
                    double y = (j + 0.5) / height;

                    // characteristic extinction coefficient, but not used for bounding
                    // higher more opaque
                    double sigma = std::exp(-2 + x * 2);

                    // the ratio of max sigma and majorant, lower more costly
                    // cannot be handled by delta tracking if larger than 1
                    double sampling_efficiency = sigma * (0.1 + y * 0.9);

                    ss[i + j * width].reset(new EstimatorType(sampling_efficiency, sigma));
                }
            }
        }
    }

    void draw()
    {
        max_cost = std::max_element(ss.begin(), ss.end(), [](std::unique_ptr<TransmittanceEstimator>& a, std::unique_ptr<TransmittanceEstimator>& b) -> bool
        {
            return a->mean_cost() < b->mean_cost();
        })->get()->mean_cost();

        double mean_error = 0;

        glBegin(GL_POINTS);
        for (int j = 0; j < height; j++)
        {
            for (int i = 0; i < width; i++)
            {
                auto& e = ss[i + j * width];
                mean_error += e->square_error();

                for (int k = 0; k < 1; k++)
                {
                    e->sample();
                }

                switch (type)
                {
                default:
                case DrawEstimate:
                {
                    double c = e->estimate_transmittance();
                    glColor3f(c, c, c);
                }
                break;
                case DrawReference:
                {
                    double c = e->reference_transmittance();
                    glColor3f(c, c, c);
                }
                break;
                case DrawCost:
                {
                    double c = e->mean_cost() / max_cost;
                    glColor3f(c, c, c);
                }
                break;
                }

                glVertex2f((i + 0.5) / width, (j + 0.5) / height);
            }
        }
        glEnd();

        mean_error /= double(width) * double(height);

        set_window_title(mean_error);
    }
};

VolumeSampler vol;

void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
    default:                                                        break;
    case ' ':       vol.info();                                     break;
    case 'r':       vol.reset<DeltaTrackingEstimator>();            break;
    case 't':       vol.reset<RatioTrackingEstimator>();            break;
    case 'f':       vol.reset<ResidualRatioTrackingEstimator>();    break;
    case 'q':       exit(0);                                        break;
    case 'e':       vol.toggle(DrawEstimate);                       break;
    case 'd':       vol.toggle(DrawReference);                      break;
    case 'c':       vol.toggle(DrawCost);                           break;
    }
}

void display()
{
    glClear(GL_COLOR_BUFFER_BIT);

    vol.draw();

    glutSwapBuffers();
    glutPostRedisplay();
}

int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitWindowPosition(1000, 0);
    glutInitWindowSize(width, height);

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutCreateWindow("");
    glDisable(GL_DEPTH_TEST);
    gluOrtho2D(0, 1, 0, 1);

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMainLoop();

    return 0;
}