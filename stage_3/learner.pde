import java.util.Arrays;

int DIMENSIONS = 4;
int[] BUCKETS = {10,40,40,20};

class Learner{
	// distance to obstacle
	// height above lower pipe
	// height below upper pipe
	// velocity
	// action

	float[][] minmax = {{10,90},	//best guess
						{-25,25},
						{-25,25},
						{-7,7}};
	float epsilon, deltaEpsilon;
	float learningRate;
	float discount;

	int replayLength = 100;
  long replays = 0;

	float[] stateCoords;
	float[] statePrimeCoords;

	NeuralNet Q, Target;

	ArrayList<Experience> memory = new ArrayList<Experience>();
	int MAX_MEMORY = 1000000;
	int MIN_MEMORY = 10000;

	int lastAction = 0;

	Learner(float deltaEpsilon, float learningRate, float discount){
		epsilon = 1;
		this.deltaEpsilon = deltaEpsilon;
		this.learningRate = learningRate;
		this.discount = discount;

		int[] layers = {DIMENSIONS,32, 32, 2};
		Q = new NeuralNet(layers, replayLength, true);
		Target = new NeuralNet(layers, replayLength, true);
	}

	void viewWorld(State s){
		if(statePrimeCoords == null){
			statePrimeCoords = stateCoords = normalise(s);	// initialise world
		}else{
			statePrimeCoords = normalise(s); 				// s' is view of the world
		}
	}

	Action act(){
		int choice;
		if(random(1) > epsilon){
			//act based on stateCoords
			double[] actions = getOutputOf(Q, stateCoords);
			choice = (actions[0] > actions[1]) ? 0 : 1;
		}else{
			choice = (random(1) > 0.5f) ? 0 : 1;
		}
		
		lastAction = choice;

		if(choice == 0){
			return Action.DOWN;
		}else{
			return Action.UP;
		}
	}

	void learn(float reward, boolean terminal){
		double[] s0, s1;
		s0 = new double[DIMENSIONS]; 
		s1 = new double[DIMENSIONS];
		for(int i = 0; i < DIMENSIONS; i++){
			s0[i] = stateCoords[i];
			s1[i] = statePrimeCoords[i];
		}
		addExperience(s0, s1, lastAction, reward, terminal);
		if(game.episodes % 2 == 0) experienceReplay();
		// prepare to act based on new state
		stateCoords = statePrimeCoords;
		// lower exploration rate
          epsilon = max(0, epsilon - deltaEpsilon);	
	}

	void experienceReplay(){
		if(memory.size() < MIN_MEMORY) return;
		int[] replayId = new int[replayLength];
		double[][] s0Matrix = new double[replayLength][DIMENSIONS];
		double[][] s1Matrix = new double[replayLength][DIMENSIONS];
		
		for(int i = 0; i < replayLength; i++){
			replayId[i] = floor(random(memory.size())); // get random experience
			Experience e = memory.get(replayId[i]);
			s0Matrix[i] = e.s0;	
			s1Matrix[i] = e.s1;	
		}

		double[][] targetMatrix;
		if(memory.size() < MIN_MEMORY + 2000){
			Q.input(s1Matrix);
			Q.feedForward();
			targetMatrix = Q.output();
		}else{
			Target.input(s1Matrix);
			Target.feedForward();
			targetMatrix = Target.output();
		}
		
		Q.input(s0Matrix);
		Q.nesterov();
		Q.feedForward();

		double[][] outputMatrix = Q.output();

		for(int i = 0; i < replayLength; i++){
			Experience e = memory.get(replayId[i]);
			double maxFutureReward = (targetMatrix[i][0] > targetMatrix[i][1])? targetMatrix[i][0] : targetMatrix[i][1];
			if(e.terminal){
				targetMatrix[i][e.action] = (double) e.reward; 
			}else{
				targetMatrix[i][e.action] = (double) e.reward + discount * maxFutureReward; 
			}
			targetMatrix[i][1-e.action] = outputMatrix[i][1-e.action];
		}

		Q.target(targetMatrix);
		double error = Q.calculateError();
		// averageError = averageError / game.episodes + error;
          averageError = averageError * (double) replays / ((double) replays + 1) + error / ((double) replays + 1);
		Q.backPropagate();

		if(replays%1500 == 0){
			Target.copy(Q);
		}
		//replays = (replays + 1) % 1500;
          replays++;
	}

	void addExperience(double[] s0, double[] s1, int action, float reward, boolean terminal){
		Experience e = new Experience(s0,s1,action,reward,terminal);
		if(memory.size() > MAX_MEMORY) memory.remove(0);
		memory.add(e);
	}

	private double[] getOutputOf(NeuralNet N, float[] inputs){
		double[][] inputMatrix = new double[replayLength][DIMENSIONS];
		for(int i = 0; i < DIMENSIONS; i++) inputMatrix[0][i] = inputs[i];	
		N.input(inputMatrix);
		N.feedForward();
		return N.output()[0];
	}

	private float[] normalise(State s){
		float[] coordinates = new float[DIMENSIONS];
		for(int i = 0; i < DIMENSIONS; i++){
			float coord = s.getDimension(i);
			
			// update minmax if exceeded
			if(coord < minmax[i][0]) minmax[i][0] = coord;
			if(coord > minmax[i][1]) minmax[i][1] = coord;
			// normalise coord
			coord = (coord - minmax[i][0]) / (minmax[i][1] - minmax[i][0]);
			coord -= 0.5;
			coordinates[i] = coord;
		}
		return coordinates;
	}
}

class Experience {
  double[] s0, s1;
  int action;
  float reward;
  boolean terminal;
  
  Experience(double[] s0, double[] s1, int action, float reward, boolean terminal){
  	this.s0 = s0;
  	this.s1 = s1;
  	this.action = action;
  	this.reward = reward;
  	this.terminal = terminal;
  }
}