# sNNake

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

## Methods

### Software
While the original AI for Snake Game repo was incredibly successful in training agents to play SS, unfortunately the learning paradigm was not ideal for our project for a few reasons:

* The entire repo was built from scratch, meaning it would have been highly impractical for us to experiment with different neural networks, or really to dynamically alter anything about the learning paradigm.
* As we dug deeper into the repo, we realized that the learning paradigm that had been implemented by scratch was highly niche and non-standard. Specifically, we realized that the NN component was not programmed to be updated via backpropagation, rather the learning paradigm initializes many networks each corresponding to a game, uses forward propagation on each to get actions for each agent, records performance metrics for each network, then creates a new “generation” of networks using a genetic algorithm which operates on those that have just been tested. Not only could we not utilize this learning paradigm to implement a reinforcement learning algorithm, we couldn’t even apply our knowledge of neural networks to parse the learning paradigm since it relied so heavily on genetic algorithms, a topic we have not formally seen in any class.
* We were interested in leveraging reinforcement learning to train snake agents and given our relative inexpertise in reinforcement learning and the difficulty of gaining expertise, it would have been infeasible to implement a reinforcement learning algorithm from scratch in the way that AI for Snake Game implemented its learning paradigm.

As a result, we decided to move in a new direction: we will use the existing SS game infrastructure from AI for Snake Game along with OpenAI Gym, a python toolkit which supplies reinforcement learning for general tasks. We acknowledge that the intimate understanding of reinforcement algorithms that would be necessary to implement this from scratch is far beyond us at the moment, instead utilizing the OpenAI Gym toolkit will allow us to experiment with reinforcement learning algorithms at a high level. So we will isolate the SS game infrastructure, then apply different OpenAI Gym reinforcement learning algorithms to train snake agents and study their behavior.

### Datasets
The nature of our project is such that we aren’t actually utilizing large datasets to train snake agents, rather we are utilizing the data created by snake agents as they play, along with a set of rewards and punishments to teach our agents the desired behavior.

### Analysis
In terms of analysis and assessment, we have altered the original AI for Snake Game repository to provide us with cleaned data describing the performance of the genetic algorthm’s training. The performance data included highscore, average score, and median score for each generation, as well as metrics for the fitness of the network. Our updated repo leveraging OpenAI Gym for reinforcement learning includes a script which trains different snake agents and also provides performance metrics. The performance metrics given here are simply high score, average score, and median score.

## Discussion
Over the course of the project we ran [BLANK number] experiments in order to improve the overall performance of the snake. Each experiment changed one or more aspects of how the training worked. In the following sections we explain our learnings through these experiments. Then we will describe the final parameters that we used to achieve a high score of [BLANK] and median score of [BLANK].

### Board Size
We experimented with two different board sizes: a 10x10 board and a 5x5 board. As one might expect, the snake was able to learn signicantly more quickly on a smaller board. On a 5x5 board, there are 25 pixels that make up the board representation. On a 10x10 board, there are 100 pixels. Thus on a 10x10 board, the snake needs much more time to really get a handle on how the game works. It takes longer for the snake to randomly find a fruit and it takes longer for the snake to randomly run into a wall.

#### Avg Scores across different board sizes

|       | Avg Score | Avg High Score |
|-------|-----------|----------------|
| 5x5   | BLANK     | BLANK          |
| 10x10 | BLANK     | BLANK          |

Note that for all data 10 million train steps and 100,000 test steps were used

### Board Representation
We had two different ways of representing the board to the RL algorithm. 

Border Not Represented             |  Border Is Represented 
:-------------------------:|:-------------------------:
![Border Not Represented](https://user-images.githubusercontent.com/19896216/144362121-37ca6e39-2698-4f48-95cb-16f3c3a31541.png)  |  ![Border Represented](https://user-images.githubusercontent.com/19896216/144362184-c0061707-abb6-443e-83f6-a9c8ac35f123.png)

To test which representation worked better, we ran 9 experiments where the border was **not** represented ([raw results](https://docs.google.com/spreadsheets/d/1z6CgToOEh4_5flIjP7yJuhiC7Jk4EMjYcKxHdy7ojSc/edit?usp=sharing))followed by the same 9 experiments where the border **was** represented ([raw results](https://docs.google.com/spreadsheets/d/13lIOnKPxrhoSfVgDq7u9D8hhewXLtbik5OmuYL_IKBg/edit?usp=sharing)). We then took the average of the mean scores and high scores for each experiment to create the following table: 

#### Avg Scores across different board representations

|                        | Mean Score | Median Score | High Score |
|------------------------|------------|--------------|------------|
| border represented     | 3.769      | 3.667        | 12.444     |
| border not represented | 3.624      | 3.667        | 11.889     |

The border representation resulted in a slight improvement to the mean and high scores. More data would be required to determine if this is statistically significant, but it makes sense that the border would help the model. Without the border representation, the snake head simply disapears when it collides with a wall. With the border representation, the snake head replaces one of the border spaces. If our hypothesis is correct, this allows the model to get a more accurate understanding of the state at which the snake fails.

### Reward Structure
[BLANK] This section is not worth writing until we actually have data

### Algorithm
Using [stable-baseline3](https://stable-baselines3.readthedocs.io/en/master/#), we were quite easily able to see how different reinforcement learning algorithms affected results. We found that...[BLANK].

#### Avg Scores across different RL Algorithms

|     | Avg Score | Avg High Score |
|-----|-----------|----------------|
| A2C | BLANK     | BLANK          |
| DQN | BLANK     | BLANK          |
| PPO | BLANK     | BLANK          |

The above table shows the average scores for the different RL algorithms. Each RL algorithm ran the same set of 7 experiments and we averaged together the results.

### Best Snake
Using all of the above learnings, we found that the best snake that we could train had the following traits:
* DQN Algorithm
* [BLANK] Reward Structure
* [BLANK] Border Representation

This snake achieved a high score of [BLANK]. We even tried testing it on a larger board than it was trained on which had very positive results!

## Reflection

In terms of what we would have done differently, really our only regret is that we didn’t find OpenAI Gym earlier. The OpenAI Gym toolkit abstracted away most of the complexities of reinforcement learning, and entirely made it possible for us to experiment with reinforcement learning. Before we found OpenAI Gym, we spent too long trying to parse the intricacies of genetic algorithms to see if we could base our experiments solely on AI for Snake Game, once we concluded that this was impractical, we again spent precious time trying to get an intimate understanding of reinforcement learning when it wasn’t needed. If we had started our project with OpenAI Gym or had looked for reinforcement learning libraries earlier, we would have had much more time to explore extensions of our work.

Regarding the continuation of our work, we see four main topics we would have liked to explore in further depth: reward functions, game modifications, hyperparameter tuning, and community contributions.

While we created and tested many different reward functions, we anticipate there are a few ways we could improve the current reward functions. One isolated example is that we created a reward structure that rewarded the snake agent for reducing its Manhattan distance from the fruit and this reward structure performed very poorly, teaching the snake to only move back and forth. However, we realized recently that this reward structure could better teach the snake if there were a punishment for increasing the snake’s Manhattan distance from the fruit, this way the undesired fidgeting back and forth would not be rewarded but rather would be a net neutral state. However, while in that case we could have created a more intelligent reward structure, in general we also could have combined our existing functions to create new, more robust reward functions. We would have liked to explore both of these avenues for improvement if we had more time.

Regarding game modifications, we were hoping to have this be an area of exploration for our project with the primary modifications in mind being to add bombs and to change the shape of the board. However, there was so much coding to be done just to get our RL working and to experiment with reward functions that we could not spare the time to implement game modifications. Also, our RL controlled snake agents only recently began performing at a fairly high level, so it would not have made much sense to prematurely throw hurdles at our underperforming snake.

When researching the different RL algorithms available with OpenAI Gym, we realized that very much like the NNs we have seen in class, RL algorithms also take hyperparameters. However, there were already so many variations of reward functions and game parameters to test in our experiments that we could not rationalize trying to tweak the different RL algorithm hyperparameters. We also have a very limited understanding of RL algorithms in general, so we would not have understood many of the hyperparameters available for us to tweak. Thus, we simply used OpenAI Gym’s default RL algorithms, however in the future this would be a great area to explore to hopefully increase training performance. 

Finally, our work could provide numerous valuable contributions to the AI gaming community. By making our work highly searchable, we could aid other research groups hoping to study similar problems. It took us a while to find OpenAI Gym, however with more OpenAI Gym projects out there, maybe other groups would be able to find the toolkit faster than we did. Moreover we could try to connect our repo to the OpenAI Gym repo adding Snake as yet another example of a game able to be played with the toolkit. This would demonstrate OpenAI Gym’s vast versatility. Lastly, we could create a pull request to add our changes to the AI for Snake Game repo. We were very careful to preserve backwards compatibility with the original genetic algorithm learning paradigm, so we could give back to the repo which lended us its game infrastructure to stoke further discussion and inspire more work to be done in this area of inquiry.

## Appendix
### Genetic Algorithm Discussion
As discussed above, we did experiment briefly with using genetic algorithms to train our agent but decided to focus on Reinforcement Learning algorithms instead. That said, before completely abandoning the effort, we trained and tested [Craig Haber's genetic algorithm snake agent](https://craighaber.github.io/AI-for-Snake-Game/website_files/index.html):

![Average Score vs  Generation](https://user-images.githubusercontent.com/19896216/137426767-8fcf979b-9b71-4596-8260-bee82b7c06da.png)

The algorithm was run for 3 days 4 hours and 44 minutes on a 10x10 board. By the end of the training, the agent was getting an average score of around 10. By contrast, we trained our best performing reinforcement learning algorithm on a 10x10 for [BLANK] hours and achieved an average score of [BLANK]. 

[BLANK] Commentary on comparison
