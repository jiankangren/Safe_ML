#include <stdio.h>
#include <math.h>

float gini_index(float n_sample, float n_empty)
{
    float p1=(float)n_sample/(n_empty+n_sample);
    float p2=(float)n_empty/(n_empty+n_sample);
    float r=1-pow(p1,2)-pow(p2,2);
    return r;

}
