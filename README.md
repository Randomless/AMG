# README
This is a pure python implement of **AMG** user-friendly for my homework project in Multilevel Iterative Method.

- [x]  Implement an AMG multilevel iterative method
    - [x]  Support AMG Setup process
    - [x]  Support different Cycle types (V | W)
    - [x]  Support smoother (GS | JACOBI)
- [x]  Plot Convergence History, make comparison

## Structure

```sh
.
├── cycle.py    # Multigrid Cycle logic , Support V | W | F
├── demo.py  # main  entry
├── matrices.mat   # test example file, 2D Poisson Matrix 2500*2500 and rhs b vector
├── multigrid.py  # Multigrid Functionality
├── README.md  
├── relative_residual.png # plot of relative residual comparison between different cycle types
├── requirements.txt # pip install -r requirements.txt
└── smoother.py # smoother logic, Support GS | JACOBI
```

## Output

Run command:
```sh
python demo.py
```

Output of `demo.py`:

```sh
Load matrix from local file, data :
A.shape:  (2500, 2500)
b.shape:  (2500, 1)
A.dtype:  float64
b.dtype:  float64
Level structure generated.
Level	 A shape	 P shape 	 A Non-zero percentage
0	 (2500, 2500)	 (2500, 1250)	 0.1968
1	 (1250, 1250)	 (1250, 324)	 0.6945
2	 (324, 324)	 (324, 85)	 2.6311
3	 (85, 85)	 (85, 21)	 9.7301
4	 (21, 21)	 (21, 3)	 34.6939
5	 (3, 3)	 		 77.7778
Using cycle_type:  V
Setup time = 7.4926111698150635 (s)
Solve  time = 19.001969814300537 (s)
Performed Cycle = 12
Residual Norm ||b-A*x||_2    = 4.127772257852449e-10
len(rel_res):  12
Level structure generated.
Level	 A shape	 P shape 	 A Non-zero percentage
0	 (2500, 2500)	 (2500, 1250)	 0.1968
1	 (1250, 1250)	 (1250, 324)	 0.6945
2	 (324, 324)	 (324, 85)	 2.6311
3	 (85, 85)	 (85, 21)	 9.7301
4	 (21, 21)	 (21, 3)	 34.6939
5	 (3, 3)	 		 77.7778
Using cycle_type:  W
Setup time = 7.4027674198150635 (s)
Solve  time = 24.61756420135498 (s)
Performed Cycle = 7
Residual Norm ||b-A*x||_2    = 5.511888382009696e-10
len(rel_res):  7
Level structure generated.
Level	 A shape	 P shape 	 A Non-zero percentage
0	 (2500, 2500)	 (2500, 1250)	 0.1968
1	 (1250, 1250)	 (1250, 324)	 0.6945
2	 (324, 324)	 (324, 85)	 2.6311
3	 (85, 85)	 (85, 21)	 9.7301
4	 (21, 21)	 (21, 3)	 34.6939
5	 (3, 3)	 		 77.7778
Using cycle_type:  F
Setup time = 8.288920402526855 (s)
Solve  time = 21.11304235458374 (s)
Performed Cycle = 7
Residual Norm ||b-A*x||_2    = 4.599764552260488e-10
len(rel_res):  7


```