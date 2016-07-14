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

	int replayLength = 20;

	float[] stateCoords;
	float[] statePrimeCoords;

	NeuralNet Q, Target;

	int lastAction = 0;

	Learner(float deltaEpsilon, float learningRate, float discount){
		epsilon = 1;
		this.deltaEpsilon = deltaEpsilon;
		this.learningRate = learningRate;
		this.discount = discount;

		int[] layers = {DIMENSIONS, 10, 10, 2};
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
			double[][] inputs = new double[replayLength][DIMENSIONS];
			for(int i = 0; i < replayLength; i++)
				for(int j = 0; j < DIMENSIONS; j++)
					inputs[i][j] = 0;	

			for(int i = 0; i < DIMENSIONS; i++) inputs[0][i] = stateCoords[i];
			
			Q.input(inputs);
			Q.feedForward();

			double[] actions = Q.output()[0];
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

	void learn(float reward){
		// float previousQ = Q[stateCoords[0]][stateCoords[1]][stateCoords[2]][stateCoords[3]][lastAction];
		// float[] possibleActions = Q[statePrimeCoords[0]][statePrimeCoords[1]][statePrimeCoords[2]][statePrimeCoords[3]];
		// float maxFutureReward = max(possibleActions[0], possibleActions[1]);

		//Q update equation
		// Q[stateCoords[0]][stateCoords[1]][stateCoords[2]][stateCoords[3]][lastAction] = previousQ + learningRate * (reward + discount * maxFutureReward - previousQ);
		
		// prepare to act based on new state
		stateCoords = statePrimeCoords;

		// lower exploration rate
		epsilon = max(0, epsilon - deltaEpsilon);	
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
		}
		return coordinates;
	}
}