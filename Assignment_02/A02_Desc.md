Problem 4:

Using Pthreads, a program is implemented to simulate a Teller-Customer program for bank. In here, there are two teller, which have a shared array that can store two units of data. On the start of day, the customers arrive and form up a queue rnadomly. At each time, a customer at the front of queue inspects if there is an empty slot in the array and writes the piece of data if he/she finds one. After that one of the free teller selects the piece of data and prints it to stdout and then takes a break of 5 seconds. The data should be printed in the same order as they have been written. This program has been implemented using PThreads and mutex locks to employ parallelism and concurrency into practice.

Problem 5:
Using Pthreads, a program is written to implement four versions of matrix Multiplication.
1. Sequential version
2. Sequential optimized version
3. Parallel version
4. Parallel optimized version