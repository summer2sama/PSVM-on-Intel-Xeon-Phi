Report
======
This is a brief report on PSVM-on-Intel-Xeon-Phi project.

Time Results
============
The training datafile for test are as follows:
 +----------+------------+---------+----------------+-----------+
 |   Name   |   Source   |  class  |  training size |  feature  |
 +==========+============+=========+================+===========+
 |   a3a    |    UCI     |    2    |      3,185     |    123    |
 +----------+------------+---------+----------------+-----------+
 |   a4a    |    UCI     |    2    |      4,781     |    123    |
 +----------+------------+---------+----------------+-----------+
 |   a5a    |    UCI     |    2    |      6,414     |    123    |
 +----------+------------+---------+----------------+-----------+
 |   a6a    |    UCI     |    2    |     11,220     |    123    |
 +----------+------------+---------+----------------+-----------+
 |   a7a    |    UCI     |    2    |     16,100     |    123    |
 +----------+------------+---------+----------------+-----------+

We compare the Time consumption on whole application and the most important step, step 2.2.3 E = I + H^T * D * H, which usually take most time of the whole application but can be offloaded to Intel Xeon Phi.

For hardware configuration:
 - CPU: 24  Intel(R) Xeon(R) CPU E5-2697 v2 @ 2.70GHz
 - MIC: Intel Corporation Device 2250 (rev 11)

Results are as follows:
 +--------------------+--------------+--------------+--------------+--------------+--------------+
 |      Datafile      |     a3a      |     a4a      |     a5a      |     a6a      |     a7a      | 
 +--------------------+--------------+--------------+--------------+--------------+--------------+
 |Application Baseline| 3.148499e+00 | 1.110746e+01 | 2.778731e+01 | 2.316730e+02 | 1.160417e+03 |
 |       Time(s)      |              |              |              |              |              |
 +--------------------+--------------+--------------+--------------+--------------+--------------+
 |Application offload-| 4.362690e+00 | 8.858282e+00 | 1.881639e+01 | 1.056090e+02 | 3.214625e+02 |
 |   openmp Time(s)   |              |              |              |              |              |
 +--------------------+--------------+--------------+--------------+--------------+--------------+
 |Application Speed up|    0.722     |    1.254     |     1.477    |     2.194    |    3.610     |
 +--------------------+--------------+--------------+--------------+--------------+--------------+
 | Step2.2.3 Baseline | 2.184750e+00 | 8.106323e+00 | 2.052040e+01 | 1.723654e+02 | 9.944381e+02 |
 |       Time(s)      |              |              |              |              |              |
 +--------------------+--------------+--------------+--------------+--------------+--------------+
 | Step2.2.3 offload- | 2.508225e+00 | 4.768063e+00 | 8.588086e+00 | 3.267663e+01 | 8.043558e+01 |
 |   openmp Time(s)   |              |              |              |              |              |
 +--------------------+--------------+--------------+--------------+--------------+--------------+
 | Step2.2.3 Speed up |    0.871     |    1.700     |     2.389    |     5.275    |    12.363    |
 +--------------------+--------------+--------------+--------------+--------------+--------------+

We could see that for small size dataset a3a, the  offload-openmp time is bigger than baseline time.

For larger dataset a4a,a5a,a6a and a7a, the speed up is bigger than 1 and grows as the traing size grows.

Problems and Solutions
======================
- To make it easier to offload code and data to MIC, I changed the data structure and some code based on the original code.
- To reduce the number of times that the application calls OFFLOAD, I brought the offload call out of the inner loop.
- To reduce the time of data transmission, I make many changes to reduced the times of data transmission, removed the unnecessary data transmission, took advantage of NOCOPY technique.
- The most important factor that influence the time consumption is the openmp overhead. I increase the granularity of the work done by openMP threads to reduce the times of call a new openmp thread. Also I only allow vector based parallelization.












