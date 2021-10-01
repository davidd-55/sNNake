## sNNake: Evolutionary Algorithms at Play

![sNNek](https://github.com/jackdavidweber/cs152-project/blob/main/snake_training.gif?raw=true)

### Contributors:
- Jack Weber
- Dave Carroll
- David D'Attile

### Introduction Outline:
- **Introductory Paragraph:** For our project, we will use the videogame Snake in order to gain a better understanding of reinforcement learning, genetic algorithms, and collaboration.

- **Background Paragraph:** Reinforcement learning is commonly used in order to train agents to "play" games. This has been done in applications ranging from Chess (Deep Blue), to Go (AlphaGo), to even Snake.

- **Transition paragraph:** Our project attempts to take the simple trained Snake-playing agent a step forward by introducing different algorithms and game variations that will add complexity to the agent's behavior. We hope that these variations will provide insights into how organisms learn and evolve to compete and collaborate.

- **Details Paragraph:** Our project’s largest hurdle will be deciding the correct information and rewards to provide to the agent in order to maximize what we consider as the optimal outcome. We plan to base our project off of and expand the capabilities of an existing, open-source Snake game library written in Python.

- **Assessment Paragraph:** 
In this section, we will analyze how long it took for the agent to learn how to play Snake under various conditions. The conditions are as follows: (1) reinforcement algorithm playing Snake, (2) reinforcement algorithm evolved using genetic algorithms playing Snake, and (3) agents are forced to collaborate playing multiplayer Snake.

- **Ethics Paragraph:** Reinforcement algorithms similar to the algorithms we used in training our agent to play Snake utilize rewards mechanisms to drive their behavior. Humans define these rewards, which can often lead to potentially unexpected outcomes when we don't truly consider the ramifications of these definitions.

### Literature Review

#### Tutorial: [How to Teach AI to Play Games: Deep Reinforcement Learning](https://towardsdatascience.com/how-to-teach-an-ai-to-play-games-deep-reinforcement-learning-28f9b920440a)
This tutorial walks through how to train a neural network to play Snake using a Deep Reinforcement Learning algorithm coded in Python. The tutorial displays very promising results accomplished with little training time. However, running our own version, we were not able to replicate the results to the same standard and we experienced problems with the game display. With some troubleshooting and debugging, this could be a good project to build on top of for our project.

#### Tutorial: [AI for Snake Game](https://github.com/craighaber/AI-for-Snake-Game)
This tutorial walks through how to train a neural network to play Snake using Deep Learning while also utilizing a genetic algorithm to combine and mutate successful neural networks. This Python code base is clean and well documented. This seems like an ideal project build on top of for our project.

#### Tutorial: [SnakeAI](https://github.com/greerviau/SnakeAI)
This tutorial similarly walks through how to train a neural network to play Snake using Deep Learning and also utilizes a genetic algorithm to combine and mutate successful neural networks. This project is coded in Processing, a language our team is not experienced with for development. This seems like an unlikely project to build on but could be useful for cross referencing learning algorithms and techniques.

#### Article: [A Hybrid Algorithm Using a Genetic Algorithm and Multiagent Reinforcement Learning Heuristic to Solve the Traveling Salesman Problem](https://link.springer.com/article/10.1007/s00521-017-2880-4)
In this article, the authors explain how they use multi-agent reinforcement learning and genetic algorithms in order to solve the traveling salesman problem. This is helpful for our project since we would like to experiment with evolving reinforcement learning algorithms with genetic algorithms. We are also interested in learning about the interactions between multiple agents, or snakes in our case.

#### Article: [Autonomous Agents in Snake Game via Deep Reinforcement Learning](https://www.researchgate.net/publication/327638529_Autonomous_Agents_in_Snake_Game_via_Deep_Reinforcement_Learning)
In this paper, the authors leverage a Deep Q-Learning Network (DQN) to teach an agent how to play Snake. The researchers’ implementation uses a series of four pre-processed screenshots of the game board as inputs, 3 convolutional layers & 1 fully-connected layer (all using the ReLU activation function), and 4 outputs corresponding to the snake's movements. The authors leveraged a novel training concept they referred to as a “training gap” - this allowed the snakes to only focus on non game-ending movement immediately after successfully eating a piece of fruit, since the direction of the respawned fruit could often lead snakes to colide with their tail.
