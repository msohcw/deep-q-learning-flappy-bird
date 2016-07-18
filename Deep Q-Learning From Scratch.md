## Deep Q-Learning From Scratch

#### Tabular Q-Learning, Deep Q-Learning, and DeepMind's research papers explained

### Introduction

### Reinforcement Learning

### The Environment

The game we will try to learn will be a [Processing][processing] clone of Flappy Bird. 

It's an ideal toy environment for several reasons 

* Limited action space, *Jump* and *Don't Jump*
* Continuous reward feedback, *+1* for each pipe passed, makes the credit assignment problem less significant
* Limited world state, namely the incoming pipes, and the player's own state

The last point is important because we will **not** be implementing a key section of the DeepMind Atari work, namely using ConvNets (Convolutional Neural Networks) to learn features of the game to be fed into the Q-Network. Rather, we will engineer features to feed it, so that we can focus on how the Q-Learning section works.

To better gauge the relative strengths of the algorithms and see how the impacts of the various improvements, there will be different physics models, 

* **Move** - with no gravity
* **Jump** - a normal flappy bird clone

We'll also modify parameters to adjust the difficulty, e.g. ```gapHeight```, size of the gap between pipes.

### Tabular Q-Learning

Like neural networks, the Q-Learning algorithm has been around for a long time, since 1989. The *Q* in Q-Learning refers to a Q-function that specifies how good an action *a* is, in a given state *s*. Using an optimal Q-function, the agent can successfully navigate an environment to maximise its reward.

Q-Learning (sometimes stated as *SARSA*) is an algorithm for training towards this optimal Q-function. It boils down to a single equation for estimating Q-values based on new information:

$Q(s,a) = Q(s,a) + learningRate \times ( r + discount \times max[Q(s', a)]  - Q(s, a))$

* $s$ - the state the agent was originally in
* $a$ - the action the agent took
* $r$ - the reward the agent received
* $s'$ - the new state the agent is in

Let's break that down.

To calculate the new expected $Q(s, a)$ we find $r + discount \times max[Q(s', a)]$. That is, we not only consider the existing reward $r$, but also the maximum future reward we expect to get. The key insight here is that we can use our existing Q-function to find that in a form of dynamic programming. We iterate over all the possible actions for $s'$ and find the maximum reward we can get, $max[Q(s', a)]. We$ then multiply that by some $discount$ factor. This reflects the idea that future rewards are less valuable because they're uncertain, and also prevents the Q-values from exploding to infinity as long as $discount \lt 1$.

We take the new expected $Q(s,a)$ and then update the existing value using some $learningRate$. To understand this, look at the spectrum where

$learningRate = 0$

$ Q(s,a) = Q(s,a)$

$learningRate = 1$

$Q(s,a) = Q(s,a) + ( r + discount \times  max[Q(s', a)]  - Q(s, a))  = r + discount \times max[Q(s', a)]$

When $learningRate = 1$, we completely replace the old value with the new one. While this seems like a reasonable idea, it actually introduces a lot of instability into the system, especially if the environment is noisy. The new $Q(s,a)$ may be an outlier caused by an extremely favourable or unfavourable environment. Hence, we use e.g. a small $learningRate = 0.01$, to aggregate information over time.

However, we cannot always just pick $$max_a[Q(s,a)]$$, because (at least initially) the policy may not be perfect. This is the *exploitation-exploration* tradeoff. Sometimes we want to explore the world and try a non-optimal action if it may lead to a better outcome later on. Sometimes we want to exploit the policy we already know to try to maximise reward.

To do this we introduce an *$\epsilon$-greedy* approach, where $0\leq\epsilon\leq1$. Every iteration, we choose to explore with probability $\epsilon$ , and choose from the actions with uniform probability. The remaining $(1-\epsilon)$ of the time, we follow $Q(s,a)$. Over time we lower $\epsilon$ by some factor until $\epsilon = 0$.

Now that we (hopefully) understand how Q-Learning works, we can move on to the implementation

```java
// in the initialisation, we setup the agent to get the current state 
agent = new Learner(0.0000001f, 0.1, 0.95); // deltaEpsilon, learningRate & discount
agent.viewWorld(game.currentState());

// then in every iteration
for(int i = 0; i< frameSpeed; i++){
  game.takeAction(agent.act());         // agent takes an action 
  game.nextFrame();                     // the world updates from s to s'
  agent.viewWorld(game.currentState()); // the agent views s'
  int reward = game.points;				// and gets reward r
  if(game.terminal){
    // If the game has ended, we apply a penalty corresponding to the maximum score
    // obtained so far. In effect, as the agent improves the penalties get worse
    reward = min(-1, -1 * game.highScore);
    record(game.points); 
  }
  agent.learn(reward); // given (s, a, r, s') the agent can now learn
}
```

The ```Learner``` class looks like this

```java
int ACTIONS = 2; // number of actions
int DIMENSIONS = 4; // number of dimensions of inputs
// Here, the dimensions are 
// Distance to next pipe, Distance to upper pipe, Distance to lower pipe, Agent's velocity

int[] BUCKETS = {30,40,40,20}; // we discretize the inputs into buckets

class Learner{
    // lookup table
	float[][][][][] Q = new float[BUCKETS[0]][BUCKETS[1]][BUCKETS[2]][BUCKETS[3]][ACTIONS];
	// minimum and maximum to normalise (feature scaling actually) inputs
    float[][] minmax = {{10,90}, {-25,25},{-25,25},{-7,7}};
  
	float epsilon, deltaEpsilon;
	float learningRate;
	float discount;

	int[] stateCoords;      // s
	int[] statePrimeCoords; // s'
	int lastAction = 0;     // a

	// constructor omitted
}
```
The ```snap``` function discretizes the continuous values of the state. Increasing or decreasing the number of buckets for each dimension affects the effective resolution for each dimension.
```java
private int[] snap(State s){
	int[] coordinates = new int[DIMENSIONS];
	for(int i = 0; i < DIMENSIONS; i++){
		float coord = s.getDimension(i);
		// update minmax if exceeded
		if(coord < minmax[i][0]) minmax[i][0] = coord;
		if(coord > minmax[i][1]) minmax[i][1] = coord;
		// normalise coord by min-max scaling
		coord = (coord - minmax[i][0]) / (minmax[i][1] - minmax[i][0]);
		coordinates[i] = round(coord * ((float) BUCKETS[i] - 1));
	}
	return coordinates;
}
```
The ```viewWorld``` function simply updates the agent's ```statePrimeCoords```. 
```java
void viewWorld(State s){
  if(statePrimeCoords == null) stateCoords = snap(s); // initialise agent
  statePrimeCoords = snap(s); 
}
```
The ```act``` function uses an $\epsilon$-greedy approach to pick actions.
```java
Action act(){
  int choice;
  if(random(1) > epsilon){ 	// act based on state
    float[] actions = Q[stateCoords[0]][stateCoords[1]][stateCoords[2]][stateCoords[3]];
    choice = (actions[0] > actions[1]) ? 0 : 1;
  }else{					// pick an action at random
    choice = (random(1) > 0.5f) ? 0 : 1;
  }
  lastAction = choice; 		// store the action for later
 
  if(choice == 0){
    return Action.DOWN;
  }else{
    return Action.UP;
  }
}
```
Now we get to the meat of the algorithm, the ```learn``` function, which essentially implements the value update equation mentioned above.
```java
void learn(float reward){
  // Q(s,a)
  float previousQ = Q[stateCoords[0]][stateCoords[1]][stateCoords[2]][stateCoords[3]][lastAction];
  // Q(s'a) for all a
  float[] possibleActions = Q[statePrimeCoords[0]][statePrimeCoords[1]][statePrimeCoords[2]][statePrimeCoords[3]];
  // max Q(s'a)
  float maxFutureReward = max(possibleActions[0], possibleActions[1]);

  //Q update equation
  Q[stateCoords[0]][stateCoords[1]][stateCoords[2]][stateCoords[3]][lastAction] = previousQ + learningRate * (reward + discount * maxFutureReward - previousQ);

  // prepare to act based on new state
  stateCoords = statePrimeCoords;

  // lower exploration rate
  epsilon = max(0, epsilon - deltaEpsilon);	
}
```

And that's it! The full code for this stage is available [here][stage_2]. You should be able to get fairly impressive results in under a few minutes of training (press ```UP``` to increase the frame speed), and the agent should play nearly-perfectly.

### Deep Q-Learning

#### Neural Nets as a Q-function approximator

In the tabular version we used a lookup table as the Q function. But there's some significant downsides to this solution.

*     There's limited generalisability. That is if we have a new state $y$ that differs from seen state $x$ by some small distance e.g. 1 indice, the agent doesn't learn from $x$ at all. That is, it doesn't generalise well to previously unseen states.

*     The limiting factor is memory. A lookup table for a state like the Atari 2600 pixel frame would be impossibly large to store.

With Deep Q-Learning, we replace the lookup table for the Q-function with a neural net to approximate the Q-function. Specifically, we have a neural net which takes in the inputs which describe the state (note that we only describe the state, and not the action), and have it produce an array of outputs describing the Q-values for all the actions. This saves us from performing a forward pass for each action, as compared to a system where we describe the action and the state as inputs and produce a single Q-value as the output.

However, since we only ever update one Q-value, $Q(s,a)$, we can only correct one of the outputs of the neural net during backpropagation. So, when we set the target values for the neural net, we update one and leave the other the same. Hence only one output node 'learns' because the other output nodes have $error = 0$.

Specifically, we 


1.                                 Input s and feed forward, to produce an array $Q(s,a)$
2.                                 Choose the best action $a*$ based on $max Q(s,a)$
3.                                 Input $s'$ and feed forward to produce an $array Q(s',a)$
4.                                 Find the maximum reward from $Q(s',a)$
5.                                 Update the entry for $a*$ in $Q(s, a)$ and use that as the target to backpropagate

#### Experience Replay

If we were to directly train the system like this, using stochastic gradient descent, we would find the system to be extremely unstable and with a tendency to diverge (i.e. not find any solution). The reason for this is *catastrophic forgetting*. 

For stochastic gradient descent to work, there is an assumption that the underlying distribution of training examples is roughly the same over time. Obviously, with online learning that assumption is definitely false, as we introduce new training examples into the training set over time. However, that can push the neural net towards one extreme to learn one behaviour, and saturate the training set with examples that result from that new behaviour. The neural net then rises to the occasion to solve the new examples, while forgetting the old examples that led to the first behaviour. 

The proposed solution to this is **experience replay** with mini-batch gradient descent. We store every experience, where an experience consists of $(s, a, r, s')$ in a memory. That is, instead of learning from the new experience every time, we insert the new experience into the memory. Then, we replay a subset, or mini-batch, of these examples to train the neural net. This ensures that both old and new experiences are trained on, and helps to even out the training distribution over time.

#### Target Net

However, as the targets are generated using the same net, there is a high likelihood that after updating $Q(s,a)$, $Q(s',a)$ will shift in the same direction. An increase in $Q(s,a)$ would lead to an increase in $max Q(s',a)$, and amplify the effects of each update. This tends to lead to instability in the algorithm and oscillation in its behaviour. 

A key innovation from the Deep Q-Learning paper was the idea of having a separate net to generate target values and update it after some delay. That is, instead of using the main net to calculate the future rewards, we use the target net, and then copy the weights from the main net to the target net every $T$ steps, where $T$ is some significant delay. This increases the stability of the algorithm as the targets in backpropagation do not shift as wildly as they might if using the same net.

Specifically, we 
1. Select a mini-batch of experiences
2. Input $s$ into $Q$ and feed forward to produce an array $Q(s,a)$
3. Choose the best action $a*$ based on $max Q(s,a)$
4. Input $s'$ into Target and feed forward to produce an $array Target(s',a)$
5. Find the maximum reward from $Target(s',a)$
6. Update the entry for $a*$ in $Q(s, a)$ and use that as the target to backpropagate
7. Copy Q to Target every $T$ steps

The toy environment that we are playing with is not really a good candidate for Deep Q-Learning, mainly because it's so trivial. However we'll press on and to look at how it can work regardless. The full starter code is [here][pre_stage_3] and includes an implementation of a regression neural net with Nesterov momentum and RMSProp.

```java
int replayLength = 100;
long replays = 0;

NeuralNet Q, Target;

int[] layers = {DIMENSIONS, 32, 32, ACTIONS}; // 2 hidden layers of 32 units
Q = new NeuralNet(layers, replayLength, true);
Target = new NeuralNet(layers, replayLength, true);
```

```java
// in act() we replace the table lookup
// float[] actions = Q[stateCoords[0]][stateCoords[1]][stateCoords[2]][stateCoords[3]];
// ... with
double[] actions = getOutputOf(Q, stateCoords);
// ...

private double[] getOutputOf(NeuralNet N, float[] inputs){
  // these lines transform the input into a minibatch matrix with 0s everywhere
  // except at the i = 0, the inputs
  double[][] inputMatrix = new double[replayLength][DIMENSIONS];
  for(int i = 0; i < DIMENSIONS; i++) inputMatrix[0][i] = inputs[i];	
  
  N.input(inputMatrix);
  N.feedForward();
  return N.output()[0]; // and return only the outputs from the inputs
}
```

We store each transition as an ```Experience```, with $s,a,r,s'$ and a boolean indicating if it is a terminal state.

```java
ArrayList<Experience> memory = new ArrayList<Experience>();
int MAX_MEMORY = 1000000;
int MIN_MEMORY = 40000; // the agent collects MIN_MEMORY before starting experience replay
  
class Experience {
  double[] s0, s1;
  int action;
  float reward;
  boolean terminal;
  // constructor omitted
}
```

The learn function is dramatically simplified. The new experience is stored and ```experienceReplay``` is executed every 2 frames (to reduce computation).

```java
// in learn() we replace the Q update function with
// ...
addExperience(s0, s1, lastAction, reward, terminal);
if(game.episodes % 2 == 0) experienceReplay()
// ...

void addExperience(double[] s0, double[] s1, int action, float reward, boolean terminal){
  Experience e = new Experience(s0,s1,action,reward,terminal);
  if(memory.size() > MAX_MEMORY) memory.remove(0); // remove oldest
  memory.add(e);
}
```

 The bulk of the work is done in the ```experienceReplay``` function

### Double Q-Learning and Double DQN

If we expand the Q-learning equation, the equation for calculating targets/updates looks like this:

$maxFutureReward$

$ = r + discount \times max[Q(s', a)] $

$= r + discount \times max[r + discount \times max[Q(s'', a)]]$

Over time, the max operator tends to cause an overestimation of how good any given state is.

To improve upon this, an original innovation in Q-Learning, Double Q-Learning, was to have two lookup tables and randomly assign experiences to each lookup table. Then, instead of using the same lookup table for calculating updates, we use one two pick the best future action and one to calculate the value of that action.

$maxFutureReward = r + discount \times max[Q_1( s', max_a Q_2(s', a) )]$

We can apply this in Deep Q-Learning by using two nets. But, we already have two nets, the Q net and the Target net, hurrah! With Double DQN, we use the Q net to pick the best action, and the Target net to estimate the value of that action.

**code**

### Prioritized Experience Replay

With experience replay, we selected experiences using a uniform probability distribution, each experience was equally likely to be picked. However, it would be sensible to instead pick experiences from which we are likely to learn the most. That's the essence of prioritised experince replay.

The question then is, how do we find experiences which we can learn most from, without actually learning them and seeing the change? The suggested method is to use the Temporal Difference (TD) error as a proxy to learning where 

$TD\ Error = |Q(s,a) - (r + maxQ(s',a))|$

Intuitively, this is a measure of how 'surprising' any given transition is. For example, the transition from $0$ reward to $1$ reward when passing a pipe would have a high $TD\ Error$ and should be replayed more often. Similarly with the transition when hitting a pipe and a $-1$  penalty results.

### Proportional Prioritization and Rank-based Prioritization

Now that we have a measure of how important any experience is, we might want to greedily replay only the set of experiences with the highest $TD\ Error$. But the obvious problem with that is that initially low $TD\ Error$ experiences may *never* be replayed. Additionally, it may cause the neural net to overfit and be weaker in the long run. (You might be noticing a trend here, in that we always have to be careful not to push the system one way or the other.) The proposed solution is a **stochastic prioritization**, with either 

* proportional prioritization, based on $TD\ Error_i$
* rank based prioritization, based on rank of $TD\ Error_i$ when sorted across the memory

$$P(i) = \frac{p_i^a}{\sum_{k} p_k^a}, p_i = |TD\ Error_i|$$ if proportional, $$p_i = \frac{1}{rank(TD\ Error_i)}$$ if rank based.

$a$ is a priority factor, where $a=0$  is no prioritization, i.e. drawing from a uniform distribution.