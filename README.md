# 3Q4F
<img src="https://github.com/Karthikay2002/3Q4F/assets/54672017/98cbe72e-a6a6-462c-bf9a-6b90cbc6e48b" alt="Image" width="400">

- GitHub Repo for Hacklytics'24 - GT Hackathon! 
- We are a team of 4, building 3Q4F (**Q**uant, **Q**uantum, and **Q**-Learning for **F**inance), a quantum algorithm-based trading platform, offering superior performance compared to Reinforcement Learning (RL) methods, revolutionizing stock trading. 

# Inspiration:-
- Classical trading methods often suffer from limited optimization capabilities and slower decision-making processes, leading to suboptimal results in dynamic market environments. Reinforcement learning (RL) was introduced to address these shortcomings by enabling sequential decision-making and adaptive strategies based on technical indicators.
 <img width="977" alt="image" src="https://github.com/Karthikay2002/3Q4F/assets/54672017/bc6bb431-0a9e-46dc-aa65-abc2452d9d86">

  However, RL algorithms may struggle with scalability, convergence issues, and susceptibility to noise, prompting the exploration of quantum computing (QC) for its potential to offer faster computation, improved optimization, and greater robustness in stock trading applications.
  
 - Taking inspiration from [[1]([url](https://arxiv.org/pdf/0810.3828.pdf))] where they theoriesed about Quantum Superposition States and Circuits for the action space, we built a Quantum Reinforcement Learning based Trader as an Enhancement of an RL based trader.
- There is theoretical evidence on the quadratic speedup for convergence of the RL algorithm on employing Quantum Computing [[2]([url](https://journals.aps.org/prx/pdf/10.1103/PhysRevX.4.031002))]

# TL;DR:-
- Classical trading methods are slow and lack optimization, prompting the introduction of Reinforcement learning (RL) for adaptive strategies. But RL faces scalability and noise issues. Enter Quantum Computing (QC), offering faster computation and better optimization, potentially revolutionizing stock trading.

# Problem Statement and Proposed Solution:-
- Problem Statement: Providing a Reinforcement Learning Based Trader that has sound performance, faster convergence, and better balance of exploration and exploitation to cover all sorts of strategies in a dynamic market environment.
- Proposed Solution: Quantum Reinforcement Learning (QRL) offers a solution by leveraging quantum algorithms to help the Reinforcement Learning Trader to balance the exploration vs exploitation issue and provide faster convergence for optimal buy-sell decision making.


# Tracks and Bounties we targetted:-
- Building for the Finance Track at Hacklytics'24!
  
# Development Flow:-
 1. **Understanding and Preprocessing Data**: Collect historical stock price data and select relevant technical indicators, like Bollinger Bands, EMA and RSI.
 2. **Implementing Quantum Reinforcement Learner (QRL)**:
     1. With the quantum states represent the value of the technical indicators and the actions are buy,sell and hold.
     2. Implementing Grover's iteration algorithm for search of optimal buy-sell signals on the state action space.
 4. **Training and Evaluation**: Training on historical price data and backtesting our strategy to assess portfolio returns, risk and performance regarding benchmark.
 5. **Portfolio Return Assessment**: Providing a framework for future training on a portfolio of stocks, providing graphs and metrics to assesss performance of our trader when compared to benchmark and a typical Q-Learning Trader.

# Challenges we ran into:-
- Trading on a portfolio of stocks, and how to diversify it for optimal conditions.

# How to solve it:-
Two options seemed viable:
- Equally splitting the portfolio among the 10 stocks (example), and optimizing the trade stocks using its own QRL based trading methods.
- Performing portfolio optimization to reduce risk for higher return, and rebalnacing portfolios periodically.
   
# Future Challenges to deal with:-
- Significant fluctuations in asset allocation due to the quantum computing-based portfolio optimizer's operation.
- Execution of substantial trade volumes leading to heightened trading activity.
- Adverse market impact, especially pronounced for less liquid assets or during periods of heightened volatility.

# Preventive Measures:-
- Implementation of transaction costs as a prudent measure to address challenges.
- Imposition of constraints on portfolio turnover to mitigate the frequency and scale of allocation adjustments.

# Benefits of Interventions:-
- Mitigation of the risk of excessive trading costs.
- Promotion of stability in portfolio management practices.
- Enhancement of overall investment efficacy.

# Images
<img width="1009" alt="Screenshot 2024-02-10 at 11 59 21 PM" src="https://github.com/Karthikay2002/3Q4F/assets/54672017/8584e092-e77c-4caf-9300-b3ae87059eb5">


![f0b6afbd62c8458698323e6d7e286e9c](https://github.com/Karthikay2002/3Q4F/assets/54672017/b5e9451a-4442-47e1-8e13-ca5f06d0becd)
![Figure1](https://github.com/Karthikay2002/3Q4F/assets/54672017/04765999-309c-419b-a6ca-b56b1b530ba0)

![7795adb5a8d540c9a59c564163f017e7](https://github.com/Karthikay2002/3Q4F/assets/54672017/4ae693bd-532e-4c84-83c2-08b2196bfd5a) ![8eeb9a37ab3048a0acfb1b6bd83ac331](https://github.com/Karthikay2002/3Q4F/assets/54672017/4b62522b-9a8c-4689-81b3-a8f0c8a2de5b)





# References
- [1] D. Dong, C. Chen, H. Li and T. -J. Tarn, "Quantum Reinforcement Learning," in IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), vol. 38, no. 5, pp. 1207-1220, Oct. 2008, doi: 10.1109/TSMCB.2008.925743
- [2] Giuseppe Davide Paparo, Vedran Dunjko, Adi Makmal, Miguel Angel Martin-Delgado, and Hans J. Briegel, Quantum Speedup for Active Learning Agents, Phys. Rev. X 4, 031002
