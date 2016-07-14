import org.ejml.simple.*;
import org.ejml.data.*;
import java.util.Random;

ArrayList<Datum> data = new ArrayList<Datum>();

//hyperparameters
enum Layer {
  INPUT, HIDDEN, OUTPUT
}

enum Activation {
  SIGMOID, LINEAR, RELU
}

class NeuralNet {
  int[] layers = new int[100];
  
  Activation[] A = new Activation[1000];
  Layer[] L = new Layer[1000]; 
  float[][] N = new float[1000][10];
  float[][] D = new float[100000][5];
  
  SimpleMatrix[] W = new SimpleMatrix[10];
  SimpleMatrix[] a = new SimpleMatrix[10];
  SimpleMatrix[] z = new SimpleMatrix[10];
  SimpleMatrix[] g = new SimpleMatrix[10];
  SimpleMatrix[] momentum = new SimpleMatrix[10];
  SimpleMatrix[] epsilon = new SimpleMatrix[10];
  SimpleMatrix[] delta = new SimpleMatrix[10];
  SimpleMatrix y;
  
  float initVariance = 0.05;
  double learningRate = 0.00003;
  double momentumSize = 0.9;
  double momentDecay = 0.9;

  int batchSize;

  int ctNeurons = 0;
  int ctDendrites = 0;
  int ctLayers = 0;
  int outputIndex;

  NeuralNet(int[] layerSizes, int batchSize, boolean regression){
    addLayer(layerSizes[0], Layer.INPUT, Activation.RELU);
    for(int i = 1; i < layerSizes.length - 1; i++) addLayer(layerSizes[i], Layer.HIDDEN, Activation.RELU);
    if(regression){
      addLayer(layerSizes[layerSizes.length - 1], Layer.OUTPUT, Activation.LINEAR);
    }else{
      addLayer(layerSizes[layerSizes.length - 1], Layer.OUTPUT, Activation.RELU);
    }
    connectLayers();
    this.batchSize = batchSize;
  }

  void addLayer(int size, Layer l, Activation fn){
    layers[ctLayers] = size;
    if(l == Layer.HIDDEN || l == Layer.INPUT) layers[ctLayers]++; // include bias term
    
    A[ctLayers] = fn;
    a[ctLayers] = new SimpleMatrix(size,batchSize);
    z[ctLayers] = new SimpleMatrix(size,batchSize);
    ctLayers++;
  }

  void connectLayers(){ 
    for(int i = 1; i < ctLayers; i++){
      int next = (i != ctLayers - 1) ? layers[i] - 1: layers[i]; //account for bias term 
      W[i] = SimpleMatrix.random(next,layers[i-1],-initVariance,initVariance,new Random());
      momentum[i] = new SimpleMatrix(next,layers[i-1]);
      g[i] = new SimpleMatrix(next,batchSize);
      epsilon[i] = g[i].copy().elementPower(0).scale(0.000001); // tiny epsilon for RMSProp
    }
  }
  
  void input(double[][] inputs){
    a[0] = new SimpleMatrix(inputs).transpose();
    a[0] = a[0].combine(SimpleMatrix.END,0, ones(batchSize));
  }

  double[][] output(){
    double[][] outputs = new double[batchSize][layers[ctLayers-1]];
    for(int i = 0; i < layers[ctLayers-1]; i++){
      for(int j = 0; j < batchSize; j++){
        outputs[j][i] = (float) a[ctLayers-1].get(i, j);
      }
    }
    return outputs;
  }

  void target(double[][] targets){
    y = new SimpleMatrix(targets).transpose();
  }
  
  void correctTargetsWith(double[][] targets){
    y = a[ctLayers - 1].copy();
    for(int i = 0; i < batchSize; i++){
      y.set((int)targets[i][0],i,targets[i][1]);
    }
  }
  
  void feedForward(){
    for(int i = 1; i < ctLayers; i++){
      z[i] = W[i].mult(a[i-1]);
      if(A[i] == Activation.RELU){
        a[i] = z[i].elementExp().plus(1.0).elementLog();
      }else{
        a[i] = z[i].copy();
      }
      if(i < ctLayers -1) a[i] = a[i].combine(SimpleMatrix.END, 0, ones(batchSize));
    }
  }
  
  float calculateError(){
    delta[ctLayers - 1] = (y.minus(a[ctLayers - 1])).negative(); // assumes linear, so f'(x) is 1
    return (float) y.minus(a[ctLayers - 1]).elementPower(2.0).elementSum() / batchSize;
  }
  
  SimpleMatrix errorMatrice(){
    return y.minus(a[ctLayers - 1]).elementPower(2.0);
  }

  void nesterov(){
    for(int i = ctLayers - 1 ; i > 0; i--){
      W[i] = W[i].minus(momentum[i].scale(momentumSize));
    }
  }

  void backPropagate(){
    for(int i = ctLayers - 2 ; i > 0; i--){
      if(A[i] == Activation.RELU){
        delta[i] = W[i+1].transpose().mult(delta[i+1]); // dNet(i+1)/dOut(i) * dE/dOut(i+1)
        delta[i] = delta[i].extractMatrix(0,delta[i].numRows()-1,0,SimpleMatrix.END); // extract from bias which has no error
        delta[i] = delta[i].elementMult(z[i].negative().elementExp().plus(1.0).elementPower(-1.0));
      }
    }
    
    for(int i = ctLayers - 1 ; i > 0; i--){
      delta[i] = delta[i].divide(batchSize).scale(learningRate).elementMult(g[i].plus(1,epsilon[i]).elementPower(-0.5)); // learningRate/sqrt(g + epsilon)
      W[i] = W[i].minus(delta[i].mult(a[i-1].transpose()).plus(momentum[i].scale(momentumSize)));
      momentum[i] = delta[i].mult(a[i-1].transpose()).plus(momentum[i].scale(momentumSize));
      g[i] = g[i].scale(momentDecay).plus(1-momentDecay,delta[i].elementPower(2)); // g' = gamma * g + (1-gamma) * delta^2
    }
  }
}

SimpleMatrix rowSum(SimpleMatrix a){
  int numRows = a.numRows();
  SimpleMatrix ret = new SimpleMatrix(numRows,1);
  for(int i = 0; i < numRows; i++) ret.set(i,0,a.extractMatrix(i,i+1,0,SimpleMatrix.END).elementSum());
  return ret;
}

SimpleMatrix ones(int l){
  double[] seq = new double[l];
  for(int i = 0 ; i < seq.length; i++) seq[i] = 1;
  SimpleMatrix ret = new SimpleMatrix(1,l);
  ret.setRow(0,0,seq);
  return ret;
}

float sigmoid(float x) { return 1.0f/(1.0f+exp(-1.0f*x)); }
float relu(float x) { return max(0,x); }
float linear(float x) { return x; }

class Datum {
  State s;
  State sP; //s prime
  float reward;
  double error;
  boolean gameOver;
  
  Datum(State _s, State _sP, float _reward, boolean _gameOver){
    this.s = new State(_s);
    this.sP = new State(_sP);
    // this.reward = min(max(_reward/maxReward,-1),1);
    this.reward = (_reward > 0) ? 0.1 : -0.1;
    this.gameOver = _gameOver;
    this.error = 100000;
  }
  
  void setError(double e){
    this.error = e;
  }
  
  @Override 
  public String toString(){
    return this.error + "";
  }
}