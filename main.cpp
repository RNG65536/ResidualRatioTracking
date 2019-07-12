// Direct implementation of the tracking methods presented in the paper
// "Residual Ratio Tracking for Estimating Attenuation in Participating Media".
// Delta Tracking : free path sampling utilizing null-collision, transmittance
// estimator is binary (whether real collision occurs in the specified range).
// Ratio Tracking : the same sampling as Delta Tracking, but records the
// null-to-majorant ratio along the path, which are multiplied to form the
// transmittance estimator
// Residual Ratio Tracking : almost the same as Ratio Tracking, but subtracts
// from a constant control density, which is analytically integrable

#include <GL/freeglut.h>
#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

using glm::vec2;
using std::cout;
using std::endl;

std::uniform_real_distribution<double> dist(0.0, 1.0);
std::mt19937_64                        rng;
double                                 randf()
{
    return dist(rng);
}

std::string title;
void        set_window_title(double mse)
{
    std::stringstream ss;
    ss << std::setiosflags(std::ios::fixed) << std::setprecision(6);
    ss << "RMSE : " << std::sqrt(mse);
    title = ss.str();
    glutSetWindowTitle(title.c_str());
}

int width = 300;
int height = 300;

enum Profile
{
    CONSTANT,
    LINEAR,
    PARABOLIC
};

Profile profile = LINEAR;

//////////////////////////////////////////////////////////////////////////
static double max_dist = 0;
static double max_cost = 0;

double sample_free_path(double sigma, double rnd)
{
    double s = -std::log(rnd + DBL_MIN) / sigma;  // log(0) -> -INF
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

class TransmittanceEstimator
{
private:
    double m_sigma;
    double m_analytic_transmittance;
    double m_sampling_efficiency;

protected:
    const double m_length = 2.0;  // traveled distance in the medium
    double       m_majorant;      // must be higher than sigma
    double       m_sigma_control;

    double   m_estimate_sum = 0;
    uint64_t m_cost_sum = 0;
    uint64_t m_num_samples = 0;

public:
    TransmittanceEstimator(double sampling_efficiency, double sigma)
        : m_sigma(sigma), m_sampling_efficiency(sampling_efficiency)
    {
        reset();
    }

    double sigma(double p) const
    {
        // density distribution :
        switch (profile)
        {
            default:
            case CONSTANT:
                // 1) constant in the traveled range
                return m_sigma;
                break;
            case LINEAR:
                // 2) linearly increasing in the traveled range
                return m_sigma * (1 + p / m_length);
                break;
            case PARABOLIC:
                // 3) parabolicly increasing in the traveled range
                return m_sigma * (p / m_length) * (p / m_length);
                break;
        }
    }

    void reset()
    {
        // density distribution :
        switch (profile)
        {
            default:
            case CONSTANT:
                // 1) constant
                m_analytic_transmittance = std::exp(-m_sigma * m_length);
                m_majorant = m_sigma / m_sampling_efficiency;
                m_sigma_control = m_sigma;  // min, max, average extinction
                                            // coefficients are all the same
                break;
            case LINEAR:
                // 2) linear
                m_analytic_transmittance =
                    std::exp(-m_sigma * m_length * 3.0 / 2.0);
                m_majorant = (2 * m_sigma) / m_sampling_efficiency;
                // m_sigma_control = m_sigma; // min extinction coefficient
                // m_sigma_control = 2 * m_sigma; // max extinction coefficient
                m_sigma_control =
                    1.5 *
                    m_sigma;  // average extinction coefficient (mostly optimal)
                break;
            case PARABOLIC:
                // 3) parabolic
                m_analytic_transmittance = std::exp(-m_sigma * m_length / 3.0);
                m_majorant = m_sigma / m_sampling_efficiency;
                // m_sigma_control = 0; // min extinction coefficient
                // m_sigma_control = m_sigma; // max extinction coefficient
                m_sigma_control =
                    m_sigma /
                    3.0;  // average extinction coefficient (mostly optimal)
                break;
        }

        m_estimate_sum = 0;
        m_cost_sum = 0;
        m_num_samples = 0;
    }

    virtual void sample() = 0;

    // Monte Carlo integration of the transmittance
    double estimate_transmittance() const
    {
        if (m_num_samples == 0)
        {
            return 0;
        }
        return m_estimate_sum / m_num_samples;
    }

    // analytic integration of the transmittance
    double reference_transmittance() const
    {
        return m_analytic_transmittance;
    }

    double mean_cost() const
    {
        double m = double(m_cost_sum) / m_num_samples;
        return m;
    }

    double squared_error() const
    {
        double d = estimate_transmittance() - reference_transmittance();
        return d * d;
    }
};

class DeltaTrackingEstimator : public TransmittanceEstimator
{
private:
    void sample() override
    {
        // delta tracking
        double e = 1;  // transmittance estimate
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
    void sample() override
    {
        // ratio tracking
        double e = 1;  // transmittance estimate
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
    void sample() override
    {
        // ratio tracking
        double e = 1;  // transmittance estimate
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

typedef enum
{
    DrawEstimation,
    DrawReference,
    DrawCost
} DrawType;

// each pixel is assigned an estimator for Monte Carlo intergration of its
// transmittance.
// horizontal pixels vary in characteristic sigma (scattering coefficient),
// vertical pixels vary in sampling efficiency (sigma to majorant ratio)
class VolumeSampler
{
    std::vector<std::unique_ptr<TransmittanceEstimator> > ss;
    DrawType                                              type = DrawEstimation;

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
    template <typename EstimatorType>
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

                    // characteristic extinction coefficient, but not used for
                    // bounding; higher more opaque
                    double sigma = std::exp(-2 + x * 2);

                    // the ratio of max sigma and majorant, lower more costly
                    // cannot be handled by delta tracking if larger than 1
                    // double sampling_efficiency =
                    // sigma * (0.1 + y * 0.9); // perhaps wrong
                    double sampling_efficiency = (0.1 + y * 0.9);  // FIXED

                    ss[i + j * width].reset(
                        new EstimatorType(sampling_efficiency, sigma));
                }
            }
        }
    }

    void reset()
    {
        for (auto& e : ss)
        {
            e->reset();
        }
    }

    void draw()
    {
        max_cost = std::max_element(
                       ss.begin(),
                       ss.end(),
                       [](std::unique_ptr<TransmittanceEstimator>& a,
                          std::unique_ptr<TransmittanceEstimator>& b) -> bool {
                           return a->mean_cost() < b->mean_cost();
                       })
                       ->get()
                       ->mean_cost();

        double mean_error = 0;

        glBegin(GL_POINTS);
        for (int j = 0; j < height; j++)
        {
            for (int i = 0; i < width; i++)
            {
                auto& e = ss[i + j * width];
                mean_error += e->squared_error();

                for (int k = 0; k < 1; k++)
                {
                    e->sample();
                }

                switch (type)
                {
                    default:
                    case DrawEstimation:
                    {
                        double c = e->estimate_transmittance();
                        glColor3d(c, c, c);
                    }
                    break;
                    case DrawReference:
                    {
                        double c = e->reference_transmittance();
                        glColor3d(c, c, c);
                    }
                    break;
                    case DrawCost:
                    {
                        double c = e->mean_cost() / max_cost;
                        glColor3d(c, c, c);
                    }
                    break;
                }

                glVertex2f((i + 0.5f) / width, (j + 0.5f) / height);
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
        default:
            break;
        case ' ':
            vol.info();
            break;
        case 'q':
            vol.reset<DeltaTrackingEstimator>();
            break;
        case 'w':
            vol.reset<RatioTrackingEstimator>();
            break;
        case 'e':
            vol.reset<ResidualRatioTrackingEstimator>();
            break;
        case 27:
            exit(0);
            break;
        case 'a':
            vol.toggle(DrawEstimation);
            break;
        case 's':
            vol.toggle(DrawReference);
            break;
        case 'd':
            vol.toggle(DrawCost);
            break;
        case 'z':
            profile = CONSTANT;
            vol.reset();
            break;
        case 'x':
            profile = LINEAR;
            vol.reset();
            break;
        case 'c':
            profile = PARABOLIC;
            vol.reset();
            break;
    }
}

void display()
{
    glClear(GL_COLOR_BUFFER_BIT);

    vol.draw();

    glutSwapBuffers();
    glutPostRedisplay();
}

int main(int argc, char** argv)
{
    cout << "space | volume estimator info" << endl;
    cout << "Q     | reset to delta tracking estimator" << endl;
    cout << "W     | reset to ratio tracking estimator" << endl;
    cout << "E     | reset to residual ratio tracking estimator" << endl;
    cout << "A     | visualize transmittance estimation" << endl;
    cout << "S     | visualize transmittance ground truth" << endl;
    cout << "D     | visualize cost" << endl;
    cout << "Z     | set medium profile to constant" << endl;
    cout << "X     | set medium profile to linear" << endl;
    cout << "C     | set medium profile to parabolic" << endl;
    cout << "Esc   | quit" << endl;

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