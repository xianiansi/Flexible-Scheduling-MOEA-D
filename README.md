## A Two-stage Search-enhanced Evolutionary Algorithm for an Aerospace Component Production Scheduling Problem
### Abstact
Aerospace components (ACs) are an important part of the aerospace equipment. Because of process specialty, AC production scheduling has always been seen a challenging work. In this paper, an improved dual-resource-constrained flexible job shop scheduling model is constructed to formulate the problem. The characteristics of AC manufacturing process are fully considered by analyzing the relationship among processes, machines, and workers. Meanwhile, the effect of machine type, worker skill level and worker labor efficiency are also incorporated into the model. To solve the model, we design an evolutionary strategy combined with TOPSIS and used Metropolis guidelines and perturbation operators to ameliorate the search process. A two-stage search-enhanced multi-objective evolutionary algorithm is proposed, which aims to minimize the completion time, production cost and worker load imbalance. Finally, experiments are conducted based on real AC production data. Experimental results verify the proposed algorithm is effective and competitive, which provides certain application value for relevant enterprises. 

![image](https://github.com/user-attachments/assets/77675e25-0f03-4dfe-aecd-32f8916517ad)
+ From the perspective of processes, it can be divided into machining processes and non-machining processes. Machining processes need to be completed on CNC machines or traditional machines while non-machining processes such as surface heat treatment and plating, do not occupy flexible machine resources. 
+ From the perspective of machines, CNC machines require senior-skilled workers for loading, unloading and programming, with higher machine energy costs and lower fatigue consumption of workers; traditional machines have medium energy costs and average fatigue consumption of worker; non-machining processes do not need to consider machine energy costs and require the highest fatigue consumption of workers.
+ From the perspective of workers, senior-skilled workers can operate any machines in the job shop and handle non-machining processes as well, with higher labor cost; general- skilled worker can only operate traditional machines and participate in non-machining processes, with medium labor cost; single-skilled workers can only operate non-machining processes.
+ 
![image](https://github.com/user-attachments/assets/a00678a3-19f8-4b3d-b5cf-e20d26f66fd4)

The first stage focuses on the evolution of the whole population. The superior solutions are firstly selected for crossover and mutation according to their TOPSIS score and then new chromosomes are accepted by Metropolis guidelines, which process is shown in Algorithm 1.
The second stage enhances the search based on the first stage. Firstly, perform	perturbation operators on each non- dominant solution in the current population. In order to determine the contribution of perturbation operators to the search process, weights are assigned to each operator based on their previous performance. Later the perturbation operator is selected based on the weight probability to accelerate convergence, which is shown in Algorithm 2.
The whole algorithm framework is displayed in Fig. 3.

![image](https://github.com/user-attachments/assets/3cac20ce-66f4-4d8a-965d-cf9bb97f43f6)

![image](https://github.com/user-attachments/assets/f59c6a91-295f-41cf-a9e7-da1bbac8b89c)

![image](https://github.com/user-attachments/assets/5155532c-46c9-4950-858a-17b024cbba15)

Changing the worker resources in original example, TSEA still performs best overall on three objection values (Table V), which indicates that the universality and robustness of TSEA. In addition, we can also infer the non-linear relationship between the worker number and production costs/load balancing, which is closely related to worker skills.
Therefore, selecting the scheduling results of TSEA as a reference scheduling solution, we can draw the Gantt chart (Fig.6). Fig.6(a) and Fig.6(b) respectively diaplay the scheduling plans of machines and workers. The numbers in the colored box represent the job index.

![image](https://github.com/user-attachments/assets/5f710438-26e4-4f5b-a283-5dadb52c654a)



