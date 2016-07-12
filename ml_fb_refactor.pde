int stageWidth, stageHeight;
int playerWidth, playerHeight;
FlappyBird game;

boolean humanPlayer = false;

void setup(){
  size(300, 200);
  stageWidth = 300;
  stageHeight = 200;

  game = new FlappyBird(PhysicsModel.MOVE);
}

void draw(){
  if(!humanPlayer) ;
  game.nextFrame();
}

// key interaction

void keyReleased() {
  if(humanPlayer){
    if(keyCode == UP) game.takeAction(Actions.UP);
    if(keyCode == DOWN) game.takeAction(Actions.DOWN);
  }
}

// import java.util.Arrays;
// import java.util.Collections;
// import java.util.Comparator;
// import java.nio.ByteBuffer;
// import java.nio.FloatBuffer;

// boolean log = false;

// //int ACTION_DELAY = 6;
// //int action_count = 0;

// boolean humanPlayer = false; //if true, play the game with UP 
// boolean humanJumped = false;
// int loadRun = -1; //load data from a given run, -1 if start from clean slate
// int run = loadRun;
// //run 9
// //run12

// int stageWidth = 300;
// int playWidth = 300;
// boolean recording = false;

// // learning constants
// float alpha = 0.5;
// float discount = 0.15;
// float epsilon = 1;
// float groundDiscount = 1;
// float penaltyConstant = -100;

// // training constants
// int pretrainEpisodes = 0;
// int fastforward = 1;

// // game constants
// int playerWidth = 10;
// int playerHeight = 10;
// float jumpForce = 2;
// float gravity = 0.3;

// // training
// int episodes = 0;
// float percentageComplete = 0;

// // state
// int a = 2;   // action
// double b = 3.1; // distance to obstacle / 10
// double c = 1;  // height above obstacle / 10 + 15
// double d = 1; //90; // acceleration * 50 + 50
// double e = 4.0;  // velocity + 20

// State s = new State(0, 0, 0, 0, 0, 0);
// State sPrime = new State(0, 0, 0, 0, 0, 0);
// float[] sVals = new float[2];

// //float[][][][][] Q = new float[a][b][c][d][e];

// // game
// int tickWait = 0;
// float reward = 0;
// float lastReward = 0;
// float dist = 0;
// float penalty = 1;
// boolean gameIsOver = true;
// ArrayList<Obstacle> obstacles = new ArrayList<Obstacle>();
// boolean trainCurriculum = false;
// float curriculum = 6;

// // physics
// PVector position = new PVector(0, 200/2);
// PVector velocity = new PVector(0, 0);
// PVector acceleration = new PVector(0, 0);

// // graph
// ArrayList<Float> rewardGraph = new ArrayList<Float>();
// ArrayList<Float> rewardAverage = new ArrayList<Float>();
// float maxReward = 100;
// float graphFactor = 1;
// float median = 1;
// float displayValue = 0;
// float displayAction1 = 0;
// float displayAction2 = 0;

// int frame = 0;

// class Obstacle {
//   PVector topRight;
//   int width = 10;
//   int height;

//   Obstacle(PVector tR, int h) {
//     this.topRight = tR;
//     this.height = h;
//   }
// }

// class State {
//   float action;
//   double pos;
//   double distanceTo;
//   double height;
//   double acc;
//   double vel;

//   State(float a, float b, float c, float d, float e, float f) {
//     action = a;
//     distanceTo = b;
//     height = c;
//     acc = d;
//     vel = e;
//     pos = f;
//   }
  
//   State(State _s){
//     action = _s.action;
//     distanceTo = _s.distanceTo;
//     height = _s.height;
//     acc = _s.acc;
//     vel = _s.vel;
//     pos = _s.pos;
//   }
// }


// void setup() {
//   background(#FFFFFF);
//   size(300, 200);

//   // maxReward = 809;
//   // rewardAverage.add(185.);
//   // rewardGraph.add(185.);
//   // episodes = 200000;
//   setupDeepLearning();
//   loadState();
//   resetGame();
  
//   if (loadRun == -1) {
//     if (pretrainEpisodes == 0) {
//     } else {
//       //pretrain the system
//       while (!humanPlayer && episodes < pretrainEpisodes) {
//         act();
//         updateGameState();
//         learn();
//         resetGame();
//         reduceGraph();

//         if (pretrainEpisodes >= 10000) { 
//           float currentPercent = float(episodes)/(pretrainEpisodes/10000)/100;
//           if (episodes%(pretrainEpisodes/10000) == 0 && currentPercent>percentageComplete) {
//             percentageComplete = currentPercent;
//             //println(percentageComplete+"% done "+median+" average");
//           }
//         }
//       }
//     }
//     //saveState();
//   } else {
//     run = parseInt(loadStrings("run.conf")[0]) + 1;
//   }
// }

// void draw() {
//   if (!humanPlayer) {
//     for (int i = 0; i < fastforward; i++) {
//       boolean acted = false;
//       //if(random(1) < 0.5){
//       if(frame == 0){
//         act();
//         acted = true;
//       }else{
//         stay();
//       }
//       frame = (frame + 1) % 5;
//       updateGameState();
//       if(acted) learn();
//       resetGame();
//     }
//   } else {
//     if(!humanJumped) stay();
//     humanJumped = false;
//     updateGameState();
//     resetGame(); 
//   }

//   drawToScreen();
//   // print("E: ");
//   // print(error);
//   // print(" D: ");
//   // println(data.size());
//   if (recording) saveFrame("run-"+run+"-######.tif");
// }

// void keyPressed() {
//   if (key == CODED) {
//     if (keyCode == RIGHT) fastforward++;
//     if (keyCode == LEFT) fastforward--;
//     if (keyCode == DOWN) fastforward = 1;
//     if (keyCode == UP) {
//       if (!humanPlayer) {
//         fastforward += 1000;
//       } else {
//         //acceleration.y-=jumpForce;
//       }
//     }
//   } 
//   if (key == 's') saveState();
//   if (key == 'r') recording = !recording;
//   if (key == 'l') log = true;
//   if (key == 'w') {
//     for(int i = 1; i < QAction.ctLayers; i++) println(QAction.W[i]);
//   }
//   fastforward = max(0, fastforward);
// }

// void keyReleased() {
//   log = false;
//   if(humanPlayer){
//     if(keyCode == UP) jump();
//     if(keyCode == DOWN) stay();
//     humanJumped = true;
//   }
// }

// void loadState() {
//   if (loadRun != -1) {
//     println("Loading run "+loadRun+"...");

//     String[] arrayBounds_s = loadStrings("run"+loadRun+"_arrayBounds.conf");
//     a = parseInt(arrayBounds_s[0]);
//     b = parseInt(arrayBounds_s[1]);
//     c = parseInt(arrayBounds_s[2]);
//     d = parseInt(arrayBounds_s[3]);
//     e = parseInt(arrayBounds_s[4]);

//     byte[] Qprev = loadBytes("run"+loadRun+"_statefile");
//     ByteBuffer bb = ByteBuffer.wrap(Qprev);

//     for (int i = 0; i<a; i++) {
//       for (int j = 0; j<b; j++) {
//         for (int k = 0; k<c; k++) {
//           for (int m = 0; m<d; m++) {
//             for (int n = 0; n<e; n++) {
//               //Q[i][j][k][m][n] = bb.getFloat();
//             }
//           }
//         }
//       }
//     }
//     epsilon = 0;

//     println("Run loaded.");
//   } else {
//   }
// }

// void saveState() {
//   run = parseInt(loadStrings("run.conf")[0]) + 1;
//   String[] run_s = {Integer.toString(run)};
//   saveStrings("run.conf", run_s);

//   //String[] netSize = {Integer.toString(a), Integer.toString(b), Integer.toString(c), Integer.toString(d), Integer.toString(e)};
//   //saveStrings("run"+run+"_arrayBounds.conf", arrayBounds_s);
//   String[] netSize = {"4","128","128","2"};
//   saveStrings("run"+run+"_netSize.conf", netSize);
  
//   ////save state to byte file
//   byte[] Q_b = new byte[1000000];
//   ByteBuffer Q_l = ByteBuffer.wrap(Q_b);

//   for(int i = 1; i < QAction.ctLayers; i++){
//     SimpleMatrix W = QAction.W[i];
//     for(int j = 0; j < W.numRows(); j++)
//       for(int k = 0; k < W.numCols(); k++)
//         Q_l.putDouble(W.get(j,k));
//   }
  
//   for(int i = 1; i < QValue.ctLayers; i++){
//     SimpleMatrix W = QValue.W[i];
//     for(int j = 0; j < W.numRows(); j++)
//       for(int k = 0; k < W.numCols(); k++)
//         Q_l.putDouble(W.get(j,k));
//   }
  
//   saveBytes("run"+run+"_statefile", Q_b);
// }


// void act() {
//   Obstacle bottom = (obstacles.size()>1)?obstacles.get(1): new Obstacle(new PVector(300, 150), 50);
//   s.action = 0;
//   s.distanceTo = dist;//max(0, bottom.topRight.x);  
//   s.height = bottom.topRight.y - position.y + playerHeight;
//   s.acc = acceleration.y;
//   s.vel = velocity.y;
//   s.pos = position.y;

//   //println(outputs);
//   //epsilon-greedy
//   if (prioritisedExperience.size() > MIN_EXPERIENCE && random(1) > epsilon) {
//     float[] outputs = getQ(new State(s));
//     sVals = outputs;
//     //println("FIRST");
//     if (log) print("stay " + outputs[0] + " hit " + outputs[1]);
//     if (outputs[0] < outputs[1]) { // hit better
//       if (log) println(" HIT");
//       jump();
//     }else{
//       stay();
//       if (log) println(" STAY");
//     }
//   } else {
//     if (random(1) < 0.5) { //random 
//       jump();
//     }else{
//       stay();
//     }
//   }
//   //if (log) println();

//   lastReward = reward;
// }

// void updateGameState() {
//   dist++;
//   reward++;
//   //add objects
//   tickWait--;
//   if (tickWait < 0) addObstacle(300);

//   //check intersect
//   if (position.y+playerHeight > 200) gameOver(penalty*groundDiscount);
//   for (int i = 0; i < obstacles.size(); i++) {
//     Obstacle o = obstacles.get(i);
//     if (intersectRects(o.topRight.x, o.topRight.y, o.width, o.height, position.x, position.y, playerWidth, playerHeight)) gameOver(penalty);
//   }

//   // update positions

//   //acceleration.y += gravity;
//   velocity.y += acceleration.y + gravity;
//   position.y+=velocity.y;
//   //acceleration.y = gravity * 2;
  

//   //position.y+=3;

//   if (position.y < 0) {
//     position.y = 0;
//     velocity.y = 0;
//     acceleration.y = 0;
//   }

//   for (int i = 0; i < obstacles.size(); i++) {
//     Obstacle o = obstacles.get(i);
//     o.topRight.x -= 2;
//     if (o.topRight.x+o.width <= 0) {
//       obstacles.remove(i);
//       //reward+=2*dist/100;
//       i--;
//     }
//   }
// }

// void learn() {
//   Obstacle bottom = (obstacles.size()>1)?obstacles.get(1): new Obstacle(new PVector(300, 150), 50);

//   sPrime.action = 0;
//   sPrime.distanceTo = dist;//max(0, bottom.topRight.x);  
//   sPrime.height = bottom.topRight.y - position.y + playerHeight;
//   sPrime.acc = acceleration.y;
//   sPrime.vel = velocity.y;
//   sPrime.pos = position.y;

//   Datum exp = new Datum(s, sPrime, reward, gameIsOver);
//   addExperience(exp);
//   if (skipCount == 0 && prioritisedExperience.size() > MIN_EXPERIENCE) experienceReplay();
//   skipCount = (skipCount + 1) % skipFrames;
// }

// void jump() {
//   acceleration.y = -jumpForce;// * 2; //-= jumpForce;//
//   //position.y -= jumpForce;
//   s.action = 1;
//   //if (log) print(" jumped");
// }

// void stay(){
//   acceleration.y += 0.25 * jumpForce;
//   acceleration.y = min(0,acceleration.y);
//   //position.y += jumpForce;
//   s.action = 0;
// }

// void gameOver(float p) {
//   if (gameIsOver) return;
//   gameIsOver = true; 
//   rewardGraph.add(dist);
//   if (episodes>0) {
//     float average = median + 1/float(episodes)*(reward - median); 
//     penalty = penaltyConstant;// * maxReward/average;
//     //rewardAverage.add(average);
//   } else {
//     //rewardGraph.add(dist);
//     //rewardAverage.add(reward);
//   }

//   if (reward > maxReward) maxReward = reward;
//   episodes++;
//   //reward -= maxReward * 5;  
//   reward = -1000;
//   if(episodes % 1 == 0) {
        
//   print("Episode "+episodes + " D: " + dist+ " E: " + error + " Experience: " + prioritisedExperience.size() + "  ");
//   print("stay " + sVals[0] + " hit " + sVals[1]);
//   println(" " +s.action + " led here.");
// }
//   if(episodes%10000 == 0) printWeights();
// }

// void drawToScreen() {
//   background(#FFFFFF);

//   noStroke();
//   fill(#AAAAAA);
//   rect(0, 0, 310, 200);

//   stroke(#000000);
//   line(310, 0, 310, 200);

//   noStroke();
//   fill(#CC0000);
//   rect(position.x, position.y, playerWidth, playerHeight);

//   fill(#00CC00);
//   stroke(#00AA00);
//   for (int i = 0; i < obstacles.size(); i++) {
//     Obstacle o = obstacles.get(i);
//     rect(o.topRight.x, o.topRight.y, o.width, o.height);
//   }

//   fill(#DDDDDD);

//   reduceGraph();
//   if (rewardAverage.size() > 0) graphFactor = median/100;
//   for (int i = 0; i<rewardAverage.size(); i++) {
//     stroke(#0000FF);
//     //point(310+i,200-rewardGraph.get(i)/graphFactor);
//     stroke(#FF00FF);
//     point(310+i, 200-rewardAverage.get(i)/graphFactor);
//   }

//   text("Episodes: "+episodes+" Furthest: "+maxReward+" Eps: "+epsilon, 10, 190);
//   text(displayValue, 100,100);
//   text(displayAction1, 100,120);
//   text(displayAction2, 140,120);
//   if (rewardAverage.size() > 0 && rewardGraph.size()>0) text("Speed: "+fastforward+" Median: "+median+" Last: "+rewardGraph.get(rewardGraph.size()-1), 10, 20);
// }

// //game functions

// void addObstacle(int i) {
//   int hole = round(playerHeight*curriculum + random(playerHeight*curriculum));
//   int height = round(position.y+random(-50, 50))-50;
//   if (height < playerHeight) height = playerHeight;
//   if (height + hole > 200 - playerHeight) height = 200 - playerHeight - hole;
//   Obstacle top = new Obstacle(new PVector(i, 0), height);
//   Obstacle bottom = new Obstacle(new PVector(i, height+hole), 200-height+hole);
//   obstacles.add(top);
//   obstacles.add(bottom);
//   tickWait = 50;
// }

// void resetGame() {
//     //acceleration.y = 0;
//   if (gameIsOver) {
//     reward = 0;
//     dist = 0;
//     if(prioritisedExperience.size() > MIN_EXPERIENCE){
//       epsilon -= DELTA_EPSILON;
//       if (trainCurriculum && curriculum > 6) curriculum -= 0.001;
//     }
//     epsilon = max(EPSILON_MIN, epsilon);
//     if (episodes%100000 == 0) println("Episode: "+episodes+" Average:"+median);

//     gameIsOver = false;
//     obstacles.clear();
//     addObstacle(100);
//     addObstacle(200);
//     addObstacle(300);
//     position.y = 200/2;
//     velocity.y = 0;
//   }
// }

// //utils

// boolean intersectRects(float x1, float y1, float w1, float h1, float x2, float y2, float w2, float h2) {
//   float left1 = x1;
//   float left2 = x2;
//   float right1 = x1 + w1;
//   float right2 = x2 + w2;

//   float top1 = y1;
//   float top2 = y2;
//   float btm1 = y1 + h1;
//   float btm2 = y2 + h2;

//   boolean xhit = false;
//   boolean yhit = false;

//   // overlap
//   if (x1 <= x2 && right1 > left2) xhit = true;
//   if (x1 > x2 && left1 < right2) xhit = true;

//   if (y1 <= y2 && btm1 > top2) yhit = true;
//   if (y1 > y2 && top1 < btm2) yhit = true;

//   if (xhit && yhit) return true;
//   return false;
// }

// void reduceGraph() {
//   if (rewardGraph.size() > 1000) {
//     int cumulative = 0;
//     int s = rewardGraph.size();
//     int ct = 0;
//     Collections.sort(rewardGraph);
//     for (int i = 0; i<rewardGraph.size(); i++) {
//       if (rewardGraph.size() < 1000) break;
//       ct++;
//       cumulative += rewardGraph.get(i);
//       if (ct == 1000) {
//         //rewardAverage.add(float(cumulative)/1000);
//         cumulative = 0;
//         ct = 0;
//       }
//       if (ct == 500) {
//         rewardAverage.add(rewardGraph.get(i));
//         median = rewardGraph.get(i);
//       }
//       rewardGraph.remove(i);
//       i--;
//     }

//     //for(int i = 1;i<rewardGraph.size();i+=2){
//     //   rewardGraph.remove(i);
//     //   rewardAverage.set(i, (rewardAverage.get(i-1)+rewardAverage.get(i))/2);
//     //   rewardAverage.remove(i-1);
//     //   i--;
//     // }
//   }
//   if (rewardAverage.size() > (stageWidth-playWidth)) {
//     for (int i = 0; i<rewardAverage.size()-1; i++) {
//       rewardAverage.set(i, (rewardAverage.get(i) + rewardAverage.get(i+1))/2); 
//       rewardAverage.remove(rewardAverage.get(i+1));
//     }
//   }
// }

// //state utils

// //float[] getStateAt(float a, float b, float c, float d, float e){
// //  State s = new State(a,b,c,d,e);
// //  //return getOutput(s);
// //}


// //void setStateTo(float a, float b, float c, float d, float e, float[] t){
// //  State s = new State(a,b,c,d,e);  
// //  //correctOutput(s, t, gameIsOver);  
// //}

// //float[] getState(State s){
// //  return getStateAt(s.action,s.distanceTo,s.height,s.acc,s.vel);
// //}

// //void setState(State s, float[] t){
// //  setStateTo(s.action,s.distanceTo,s.height,s.acc, s.vel, t);
// //}