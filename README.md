## Optimized radio resource allocation in 5G using Deep Q-Networks (DQN)
# Abstract
The transition from 4G to 5G has not only been marked by growth in traffic, as has been usual in the last decade, but has also been accompanied by an increase in the number of devices and services. Services like
Massive Internet of Things (mIoT), Massive Machine Type Communication (mMTC), ultra-Reliable Low Latency Communication (uRLLC), and Enhanced Mobile Broadband (eMBB) necessitate the implementation of net
work slicing, which considerably increase network complexity.

To accommodate emerging services and successfully transition to future networks, operators must tackle the challenge of managing and optimizing their network infrastructure. In this context, Machine Learning (ML) and
Artificial Intelligence (AI) emerge as promising technologies capable of enhancing the efficiency of numerous processes that have traditionally relied on the expertise of human specialists. The integration of AI into 5G and future 6G networks, while promising, presents challenges, such as training
models with the vast amounts of data generated by these networks. However, by leveraging this data, AI can detect patterns and trends that enable proactive resource allocation, load balancing, or energy-saving measures as
some application examples.

Despite the clear potential of these technologies, the adoption of AI and ML methods in mobile networks remains in its early stages. The numerous challenges that lie ahead, coupled with the certainty that this technology
will be transformative, have driven the initiation of this project. The goal is to make a contribution to the ongoing academic and industrial efforts in this field. To achieve this, a DQN-based radio resource allocation agent has been developed, designed to optimize network parameters to meet specific
service requirements within a 5G network slice.

# Project Structure
The content of the project can be found in the file _master_thesis_report.pdf_.

**Chapter 1: Introduction.** Sets the stage for the project by providing an overview of the emerging possibilities enabled by the integration of AI and ML in mobile networks. 
It outlines the motivation and objectives of the work, offering the reader a clear understanding of its purpose. Additionally, this chapter includes a section on project planning,
detailing the tasks required for the project’s execution over time.  The resource planning and the estimated total cost of this project are also included in the mentioned section.

**Chapter 2: State of the Art.** Describe all the key enablers on which this project is based. The chapter is clearly divided into three main sections. The first section offers a general overview of Fifth
Generation (5G), focusing on key radio access concepts such as spectrum and overall transmission structure. It concludes with a discussion on Network Slicing technology. The second section reviews the Self
Organizing Networks (SON) paradigm and its operation within 5G networks. This review includes an examination of the most popular AI/ML optimization methods applied to SON networks. Finally, the third section presents the theoretical foundations of Deep Reinforce
ment Learning (DRL), with a detailed exploration of Deep Q-learning Network (DQN) and the implementation keys of this algorithm.

**Chapter 3: System model and problem definition.** This chapter describes the system model adopted for this project, describing the network design. It concludes by defining the problem statement addressed within this system.

**Chapter 4: Solution Design.**  It presents the solution developed to address the use case outlined in this Master’s Thesis. First, the solution is outlined and introduced at a high level, followed by a detailed explanation of each of its components (action space, state space,
reward algorithm). Lastly, the implementation of the solution using the Gym reinforcement learning framework and the interaction with the DQN agent are described.

**Chapter 5: Experimental evaluation and results.** This chapter analyses the results obtained for the DQN agent developed in the project. The experimental setup in which the agent was tested is
detailed, including two primary experiments. The first experiment examines the impact of various parameters on the agent’s training process. The second experiment provides a step-by-step demonstration
of the agent’s performance under various conditions, highlighting the effects of its actions on network parameters.

**Chapter 6: Conclusions and future works.** At the end of this document, the conclusions drawn from the development of this work are presented. Additionally, this chapter offers insights into potential future research directions on the topic.

**Appendix A: UEs assignment to serving SCs.** This appendix outlines the procedure used in the proposed system model to assign cells to UEs. For this purpose, the implemented algorithm is presented.

**Appendix B: RAN simulator.** It describes how the network model is implemented as a RAN simulator in Python. It provides a detailed explanation of the developed pseudocode.

**Appendix C: Advantages of discretizing throughput as a state variable.** The content of this appendix relates to Chapter 4, demonstrating why the design choices made for the state space in that chapter
are optimal for facilitating the agent’s learning.

**Appendix D: Code repository.** Thisappendixdetails the structure of the code developed for this project, which is available in GitHub.
