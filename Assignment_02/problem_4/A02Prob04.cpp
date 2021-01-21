#include <iostream>
#include<vector>
#include<time.h>
#include<queue>
#include<cstdlib>
#include<pthread.h>
#include<unistd.h>

using namespace std;

int NUM_CUST=24;
const int NUM_TELLER=2;
vector<int> tokens;

struct Customer{
    int token;
};

queue<Customer> custQueue;
int sharedArr[16];
/*s denotes the starting point from which data still needs to be written
numd denotes the number of customer who have written and also are writing into the queue
numtr denotes the number of customers who have been served and also are currently being served
numwd denotes the number of customers who have written into queue but have not been served
numw denotes the number of customers who have written and are also writing into the queue but not been served yet
numr denotes the number of customers currently being served by teller
*/
int s=0, numd=0, numtr=0;
int numwd=0, numw=0, numr=0;
int lastToken = 0;
pthread_t tellers[NUM_TELLER];
pthread_t readQ[2];
pthread_mutex_t sharArrayComp[2]; //mutex for locking shared array components for read/writes on different part
pthread_mutex_t numd_m,numw_m,arrpar_m,numwd_m,numtr_m,numr_m;
pthread_mutex_t competeQ;
pthread_mutex_t tokengen;


int genToken(){
	pthread_mutex_lock(&tokengen);
	int randn = rand()%100;
	lastToken+= (1 + randn);
	int rettok = lastToken;
	pthread_mutex_unlock(&tokengen);
	return rettok;
}


void *assignTokenFillQueue(void *threadint){

	int tok = genToken();
	int randn = rand()%10;
	usleep(randn * 100);  //for introducing non-determinism
	Customer ctemp;
	ctemp.token = tok;
	pthread_mutex_lock(&competeQ);
	custQueue.push(ctemp);
	cout<<" "<<tok;
	pthread_mutex_unlock(&competeQ);
}

void *tellerWork(void* threadid){
	int *telln = (int *) threadid;
	bool flag=true;
	while(flag){
		pthread_mutex_lock(&numtr_m);
		if(numtr==NUM_CUST){
			pthread_mutex_unlock(&numtr_m);
			flag = false;
		}
		else{
			pthread_mutex_lock(&numwd_m);
			pthread_mutex_lock(&numr_m);
			if(numwd==numr){
				pthread_mutex_unlock(&numwd_m);
				pthread_mutex_unlock(&numr_m);
				pthread_mutex_unlock(&numtr_m);
			}
			else{
				cout<<"Teller "<<(*telln)<<" is going to service a customer from shared array"<<endl;
				pthread_mutex_lock(&arrpar_m);
				int beg = (s + 8*numr)%16;
				numr++;
				numtr++;
				pthread_mutex_lock(&sharArrayComp[beg/8]);
				pthread_mutex_unlock(&numwd_m);
				pthread_mutex_unlock(&numr_m);
				pthread_mutex_unlock(&numtr_m);
				pthread_mutex_unlock(&arrpar_m);
				string str = "";
				for(int i=0;i<8;i++){
					str+=to_string(sharedArr[beg+i]);
					if(i!=8){
						str+="\t";
					}
				}
				while(s!=beg); //ensures that the order are written in same order as they are processed
				cout<<str<<endl;
				cout<<"Teller "<<(*telln)<<" has served a customer and will take a break of 5 seconds"<<endl;
				pthread_mutex_unlock(&sharArrayComp[beg/8]);
				pthread_mutex_lock(&numwd_m);
				pthread_mutex_lock(&numr_m);
				pthread_mutex_lock(&numw_m);
				pthread_mutex_lock(&arrpar_m);
					s= (s+8)%16;
					numw--,numwd--,numr--;
				pthread_mutex_unlock(&arrpar_m);
				pthread_mutex_unlock(&numw_m);
				pthread_mutex_unlock(&numr_m);
				pthread_mutex_unlock(&numwd_m);
				
				sleep(5);

				
			}
		}
	}
}

void *customerWork(void* arg){

	bool flag=true;
	while(flag){
		pthread_mutex_lock(&numd_m);
		if(numd==NUM_CUST){
			pthread_mutex_unlock(&numd_m);
			flag = false;
		}
		else{
			pthread_mutex_lock(&numw_m);
			if(numw==2){
				pthread_mutex_unlock(&numw_m);
				pthread_mutex_unlock(&numd_m);
			}
			else{
				Customer ctemp = custQueue.front();
				int tok = ctemp.token;
				custQueue.pop();
				numd++;
				pthread_mutex_lock(&arrpar_m);
				int beg = (s + 8*numw)%16;
				numw++;
				pthread_mutex_lock(&sharArrayComp[beg/8]);
				cout<<"Customer is going to write "<<tok<<" into the shared array"<<endl;
				pthread_mutex_unlock(&arrpar_m);
				pthread_mutex_unlock(&numw_m);
				pthread_mutex_unlock(&numd_m);
				for(int i=0;i<8;i++){
					sharedArr[i+beg] = tok;
				}
				//sleep(1);
				cout<<"Customer has written "<<tok<<" into the shared array"<<endl;
				pthread_mutex_lock(&numwd_m);
				numwd++;
				pthread_mutex_unlock(&numwd_m);
				pthread_mutex_unlock(&sharArrayComp[beg/8]);
				
			}
		}
	}

}


int main(int argc, char* argv[])
{
	
	if(argc==1 || argc>2){
		cout<<"Please enter the number of customers as argument"<<endl;
		exit(1);
	}
	
	char* ncarg = argv[1];
	NUM_CUST = stoi(ncarg);
	
	srand (time(NULL));

	//customers compete to get into queue
	cout<<"Token Order for Queue:";
	pthread_t custQueueing[NUM_CUST];
	pthread_mutex_init(&competeQ,NULL);
	pthread_mutex_init(&tokengen,NULL);
	for(int i=0;i<NUM_CUST;i++){
		pthread_create(&custQueueing[i],NULL,assignTokenFillQueue,NULL);
	}

	for(int i=0;i<NUM_CUST;i++){
		pthread_join(custQueueing[i],NULL);
	}	

	cout<<endl;

	pthread_mutex_init(&numd_m,NULL);
	pthread_mutex_init(&numw_m,NULL);
	pthread_mutex_init(&numr_m,NULL);
	pthread_mutex_init(&numwd_m,NULL);
	pthread_mutex_init(&numtr_m,NULL);
	pthread_mutex_init(&arrpar_m,NULL);
	pthread_mutex_init(&sharArrayComp[0],NULL);
	pthread_mutex_init(&sharArrayComp[1],NULL);
;
	for(int i=0;i<NUM_TELLER;i++){
		int *arg = (int *)malloc(sizeof(int));
		*arg=i;
		pthread_create(&tellers[i],NULL,tellerWork,arg);
		pthread_create(&readQ[i],NULL,customerWork,NULL);
	}

	pthread_join(tellers[0],NULL);
	pthread_join(tellers[1],NULL);
	pthread_join(readQ[0],NULL);
	pthread_join(readQ[1],NULL);

	return 0;
}


