# sNNake: Evolutionary Algorithms at Play

![sNNake](https://github.com/jackdavidweber/cs152-project/blob/main/snake_training.gif?raw=true)

## Contributors:
- Jack Weber
- Dave Carroll
- David D'Attile

## Introduction
### Background
Even if you are not interested in computer science, you have almost definitely interacted with reinforcement algorithms in your day-to-day life. These algorithms suggest which Netflix shows we will like. They help advertisers determine the optimal products to present us with. They are better than humans at [Chess](https://www.ibm.com/ibm/history/ibm100/us/en/icons/deepblue/), [Go](https://www.youtube.com/watch?v=WXuK6gekU1Y&ab_channel=DeepMind), and [driving](https://www.scientificamerican.com/article/are-autonomous-cars-really-safer-than-human-drivers/).

Given the importance of these algorithms in our lives, sNNake is a project aimed at better understanding how reinforcement algorithms work. We do this by focusing on the goal of teaching an agent to play [Screen-Snake (SS)](https://en.wikipedia.org/wiki/Snake_(video_game_genre)). SS is a simple game where the snake tries to grow in length by eating more _fruits_. The snake dies when it collides with the boundary wall or part of its own body.

So, how do we teach an agent to play SS? Well, one way to do it would be to try to come up with a set of rules that we feed directly into the agent. Rules like “if collision is imminent, turn left” or “if moving away from fruit, turn around.” This would be quite difficult and take a lot of time.

Another option is to get the agent to learn by trial and error. We let the agent play randomly and whenever it does something positive (like get a fruit) we reward it. Whenever the agent does something negative (like collide with a wall), we punish it. Over time, by reinforcing positive actions and disincentivizing negative actions, the agent will start to figure out the best strategies to get more positive rewards. The agent will learn how to play SS without us ever having to explicitly teach it.

This is the essence of reinforcement learning algorithms. The algorithm requires us to provide rewards for actions. It then tries many strategies to complete a given task. The strategies that are rewarded are given more weight and the algorithm becomes better at the assigned task.

### Details
Rather than implementing SS from scratch, we used [AI for Snake Game](https://github.com/craighaber/AI-for-Snake-Game) by Craig Haber. Built on top of pygame, this repo provided us with a basic implementation and visualization of the game. It also came with files to train and test a snake agent built with a genetic algorithm. While initially we experimented with it, we eventually concluded that the genetic algorithms were beyond the scope of our project (more details in methods section). Instead, we used [Open AI Gym](https://gym.openai.com/) for the reinforcement learning infrastructure. Figuring out how to make Haber's repo work with Open AI Gym was initially challenging, but we ended up with a system that enabled easy and rapid experimentation. Our final repo is highly parameterized, allowing easy changes of board size, reinforcement learning algorithm, reward function, and board representation.

### Assessment
On a 5x5 board, the trained agent was able to get a high score of 20 and a median score of 10. This is a pretty great result. A 5x5 board has 25 board spaces. A high score of 20 is a mere 4 points away from beating the game. In the methods section we will go into more depth on how we implemented the system capable of achieving such a result. In the discussion section we will go through the many experiments that we ran to go from a snake that continuously ran into walls to a snake that has relative mastery of the game.

### Ethics
As discussed above, reinforcement algorithms, similar to the algorithms we use in training our agent to play SS, utilize rewards mechanisms to drive behavior. Humans define these rewards, and in general, this can lead to unexpected and detrimental behavior especially when the desired behavior is very high stakes. For example, if the military decided to train a robotic agent using a reinforcement algorithm to seek out adversaries in a warzone, it would be incredibly hard to accurately define that “reward” and of course, this could potentially lead to the agent accidentally flagging civilians as adversaries. 

Thankfully, our project is quite removed from the physical world and our agent is not being trained with any real-world data; it is simply learning how to play a video game and our exploration is in the pursuit of improving our own understanding of reinforcement algorithms. Nonetheless, it is almost always the case that innocuous technologies can be reintroduced into real-world contexts, potentially leading to harsh ramifications. It is with this in mind that we would like to formally state that the research and experiments conducted in this project are in no way intended to be repurposed for harm of any kind.

Given that our project is only in its early stages--we are only at the moment experimenting with different NNs for learning to play SS--the ethical considerations are fairly generic and would apply to any project involving reinforcement algorithms. However, as our project evolves and we start to experiment with changes to the game itself, we may come across more specific ethical considerations. For example, one idea we might explore is training snake agents in different environments, in other words changing the shape and size of the board and/or introducing obstacles. Returning to our war-machine example, we would not want our success in training an obstacle-weary snake agent to translate into the development of an obstacle-weary war-machine. Or for another example, we are interested in experimenting with training multiple snake agents to play in the same game with the hopes of observing learned collaboration. We would also not want success in this area to contribute to the development of a collaborating army of war-machines!

Another completely different dimension of ethical concerns arise from considering the relevance of our project to biology. After all, our project centers around training agents to survive and adapt in an environment with particular constraints. We are interested in trying to breed successful snake agents and select behavior that leads to success. While this project borrows language from biology and loosely simulates a kind of evolution, we acknowledge that the development of snake behavior in this project does not necessarily apply to the development of human or other animal behaviors. Any non-trivial connection made between the two should excite major scrutiny.

## Related Works
For related works, our team opted to investigate both academic literature and implementations related to our project. 

In the first article we reviewed, [A Hybrid Algorithm Using a Genetic Algorithm and Multiagent Reinforcement Learning Heuristic to Solve the Traveling Salesman Problem](https://link.springer.com/article/10.1007/s00521-017-2880-4) the authors explain how they use multi-agent reinforcement learning and genetic algorithms in order to solve the traveling salesman problem. This is helpful for our project since we would like to experiment with evolving reinforcement learning algorithms with genetic algorithms. We are also interested in learning about the interactions between multiple agents, or snakes in our case.

In [Autonomous Agents in Snake Game via Deep Reinforcement Learning](https://www.researchgate.net/publication/327638529_Autonomous_Agents_in_Snake_Game_via_Deep_Reinforcement_Learning), the second paper we reviewed, the authors leverage a Deep Q-Learning Network (DQN) to teach an agent how to play SS. The researchers’ implementation uses a series of four pre-processed screenshots of the game board as inputs, 3 convolutional layers & 1 fully-connected layer (all using the ReLU activation function), and 4 outputs corresponding to the snake’s movements. The authors leveraged a novel training concept they referred to as a “training gap” - this allowed the snakes to only focus on non game-ending movement immediately after successfully eating a piece of fruit, since the direction of the respawned fruit could often lead snakes to collide with their tail.

We chose to explore open-source implementations with the goal of selecting a project to build sNNake upon. This approach allowed us to focus less on building a SS game from scratch and more on the implementation of NNs teaching agents how to play the game. 

First, we explored the [SnakeAI](https://github.com/greerviau/SnakeAI) tutorial which walks through how to train a neural network to play SS using Deep Learning and also utilizes a genetic algorithm to combine and mutate successful neural networks. This project is coded in Processing, a language our team is not experienced with for development. This seems like an unlikely project to build on but could be useful for cross referencing learning algorithms and techniques.

Second, we investigated [How to Teach AI to Play Games: Deep Reinforcement Learning](https://towardsdatascience.com/how-to-teach-an-ai-to-play-games-deep-reinforcement-learning-28f9b920440a). This tutorial walks through how to train a neural network to play Snake using a Deep Reinforcement Learning algorithm coded in Python. The tutorial displays very promising results accomplished with little training time. However, running our own version, we were not able to replicate the results to the same standard and we experienced problems with the game display. 

Finally, [AI for Snake Game](https://github.com/craighaber/AI-for-Snake-Game) walks through how to train a neural network to play SS using Deep Learning while also utilizing a genetic algorithm to combine and mutate successful neural networks. Importantly, this Python code base is clean and well documented. This seems like an ideal project to base our project on.

Among all projects surveyed, we chose to base sNNake off of AI for Snake Game for the following reasons:
* Code base is clearly commented and cleanly written
* Code base written in Python 3
* Implementation contained easily modifiable game board/logic rules
* Implementation contained a genetic NN component for training that we can study
* Implementation contained a framework for easily expanding existing NN capabilities

We believe that the combination of reviewing academic research and experimenting with pre existing applications have provided our team ample background to pursue the implementation of sNNake.

## Project Update #1

### Software Details:

As a component of the literature review, each member of our group attempted to clone and run various snake game implementations capable of training a single agent to play the game (see above). Based on our findings, we selected the implementation that best matched the following traits:

- Code base is clearly commented and cleanly written
- Code base written in Python 3
- Implementation contained easily modifiable game board/logic rules
- Implementation contained a genetic NN component for training
- Implementation contained a framework for easily expanding existing NN capabilities

With these key features in  mind, we have decided to base our project on *[AI for Snake Game](https://craighaber.github.io/AI-for-Snake-Game/website_files/index.html)* written by [Craig Haber](https://github.com/craighaber). This project is well-written, easily extensible, and we have already made modifications that allow for training to run on the Pomona server. The original repository can be found [here](https://github.com/craighaber/AI-for-Snake-Game), and our fork can be found at [sNNakeCode](https://github.com/jackdavidweber/sNNakeCode).

We will integrate PyTorch into the project when designing our own algorithms, which will provide us with the framework to try different types of algorithms using classroom techniques.

#### Data Details:

Since this project leverages the use of neuroevolution, training is initialized with a random set of parameters that are fine-tuned over the specified training course. 

Thus, we will not require an initial dataset for this project. Rather, we will generate data through simulation and experiment with how different networks perform.

#### Project Details:

As described in the Software Details section, the *AI for Snake Game* project we are building on includes a fully-funtional neuroevolution-based NN capable of training an agent to play the snake game. Our details below reflect the implementation of this network, but we fully intend to implement and compare multiple networks' performance over the course of this project. We also plan to make changes to the information and incentives that the agent is presented with.

- Type of NN(s) in Use: 
    
    This project leverages the use of a neuroevolution-based NN composed of a separate genetic algorithm (GA) and NN. This functions by training snake agents via a 4 layer (1 input, 2 hidden, 1 output) NN. 
    
    At the end of a training epoch, the set of *n* agent results is fed to a GA that improves agent performance through the following four methods (note that Fitness Calculations occur once per epoch, and Selection, Crossover, and Mutation are performed sequentially *n*/2 times):

    - Fitness Calculations:  Calculations that assign scores to each agent based on its performance during the training epoch
    - Selection: The algorithm selects 2 high-performing agents to be used for Crossover
    - Crossover: The algorithm mixes characteristics of each relatively successful parent agent to create a new child agent
    - Mutation: The algorithm randomly mutates aspects of the child agent's characteristics in order to prevent training stagnation

    After the above steps are performed, the GA creates a set consisting of the top 50% of agents from the initial population and the newly generated child population and propogates them to the following training epoch. This process repeats for a specified number of epochs that each contain gameplay for a specified number of unique agents.

    Further implementation details and the information provided here can be found on the original *AI for Snake Game* [project page](https://craighaber.github.io/AI-for-Snake-Game/website_files/index.html).

- NN Input Details:
    - The [Manhattan distance](https://xlinux.nist.gov/dads/HTML/manhattanDistance.html) between the snake agent's head and the game board fruit
    - Available space on the grid in each of the four available directions (left, right, up, down) from the snake agent's head
    - The current length of the snake agent

- NN Ouput Details: 
    - This network has four outputs (representing the four available movement directions) ranging in value from 0 to 1 where
    - The snake agent will move in the direction of the output with the highest associated value


## Project Update #2

### What have we completed/tried to complete?

We have been able to successfully modify the *AI for Snake Game* project such that we can train an agent to play the snake game via neuroevolutionary methods on the Pomona server. Previously, this was not possible since the game required a graphical representation of the training/playing. Additionally, we have added improved data collection and visualization for fitness through generations.

We have attempted to implement our own fork of the neuroevolutionary algorithm in use withing this project, however, we have realized that there is extensive research required on our part to achieve this. As a result of this recognition, we have spent the bulk of our time up until now researching methods such as Deep-Q reinforcement learning. We now plan to leverage OpenAI Gym for the implementation of various algorithms.

### What issues have we encountered?

We have encountered two major issue so far over the course of this project:

- The initial genetic algorithm was implemented in an entirely manual way; there is no use of PyTorch. This has led to the additional development overhead of integrating PyTorch into the project.
- As previously discussed, we have found that more research overhead than previously expected will be needed for the effective implementation of reinforcement learning and neuroevolutionary algorithms.

## Methods Outline

### Software
Unfortunately the NN component of the original *AI for Snake Game* was implemented from scratch making it impractically difficult for us to experiment with different networks. Also, as we dug deeper into this repository we realized that the NN component does not actually utilize reinforcement learning and trying to implement this kind of algorithm from scratch with the existing code would once again be impractical and not dynamic. As a result, we have decided to move in a new direction: we will use the existing SS game logic from *AI for Snake Game* along with OpenAI Gym to reliably and dynamically implement a reinforcement learning algorithm that will train snake agents. We acknowledge that the intimate understanding of reinforcement algorithms that would be necessary to implement this from scratch is far beyond us at the moment and instead utilizing the OpenAI Gym toolkit will allow us to experiment with reinforcement learning algorithms at a high level.

### Datasets
The nature of our project is such that we aren't actually utilizing large datasets to train snake agents, we are rather utilizing the data created by snake agents playing along with a set of rewards and punishments to teach our agents the desired behavior.

### Analysis
In terms of analysis and assessment, we have altered the original *AI for Snake Game* repository to provide us with cleaned data describing the performance of the genetic algorthm's training. For the Reinforcement Learning Algorithm, we created a notebook that allows us to train and test different versions of agents against each other.

## Discussion Outline
In this section, we will present how game and algorithm modifications affected the average high score of the snake after X hours of training and Y hours of testing.

* **RL vs. GA**: The Reinforcement Learning Algorithm generally tended to perform (better/worse) than the Genetic Algorithm. We attempted to make the comparison as fair as possible but do recognize that the differences in the way the algorithms work make a 1:1 comparison very difficult which likely introduced noise.
* **Reward Structure**: For the RL Algorithm, certain reward structures worked better than others. For example, penalizing the snake for not finding a fruit (improved/decreased) the average high score by ___%.
* **Different RL Algorithms**: [A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html) generally performed (better/worse) than [DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html) which did (better/worse) than [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
* **RL Algorithm HyperParameter Tuning**: We were able to improve average high score by ___ when we changed the learning rate, number of hidden layers, number of neurons, etc.
* **Board Size**: Decreasing the board size made it significantly harder for the snake to score well. Even if we adjust based on the high score, we still find that on smaller boards the snake does relatively worse.
* **Other Game Mods**: Adding multiple snakes had an interesting effect... Adding bombs also had an interesting effect...
