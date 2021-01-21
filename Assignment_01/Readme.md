The objective of problem P4 is to implement a cache miss analyzer for any given code file. The test cases involve loop nests with a single source loop and upto 3D array references. The Cache can be of three types - FullyAssociative, SetAssociative and DirectMapped with value of number of lines per set to be provided in case of SetAssociative and all other details to be provided as well. Check the Analysis.java file for understanding the cahce miss analysis methods. The code also utilizes g4 Grammar file and antlr to form the callbacks for Grammar. The grammar can be visualized using a testcase file and grun command.

In order to run and setup the test file say UnitTestCases.java in linux, simply copy the files onto a directory and run the script run.sh by providing UnitTestCases.java as an argument.

bash run.sh UnitTestCases.java

In order to run it on Windows, you need to execute the following commands:

1. java -Xmx500M org.antlr.v4.Tool LoopNest.g4
2. javac LoopNest*.java
3. javac Driver.java
4. java Driver \<\<TestFile\>\>

The output will be printed on stdout and will be stored in Results.obj (Serialized Object File) file in the form of List<HashMap<String,Long> >, which is a list of HashMaps storing the name of arrays against their respectve cache misses.

Note: A sample test file is given - UnitTestCases.java
