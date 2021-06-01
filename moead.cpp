#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include <values.h>
#include <float.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <algorithm>
#include <unistd.h>

/***
    AUTHOR: Marlon da Silva Rogério
    DATA: 01 JUN 2021
    DISCIPLINA: Tópicos Especiais em Inteligência Computacional 
    DESCRICAO: Otimização multiobjetivo | MOEA/D | Mimização
***/

using namespace std;

const int T = 10;                                 //tamanho da vizinhanca
long geracoes = 500;                              //numero de geracoes
int tam_pop;                                      //tamanho da populacao
const int dimensoes_obj = 2;                      //numero de objetivos
const int dimensoes_var = dimensoes_obj + 10 - 1; //numero de variaveis de decisao (valor padrao do DTLZ2)
const double prob_mutacao = 0.1;                  //probabilidade de mutacao
double limiteInferior = 0, limiteSuperior = 1;    //todas as solucoes devem estar entre 0 e 1 (padrao do DTLZ2)
double dados[1000][20];                           //considerando que vou ter no maximo 1000 vetores de peso com no maximo 20 objetivos
double ideal[dimensoes_obj];                      //vetor ideal z*
char pesos[6];                                    //usado pra facilitar a leitura dos arquivos de pesos

struct Individuo
{
    double x[dimensoes_var];
    double fx[dimensoes_obj];
    double peso[dimensoes_obj];
    int vizinhanca[T]; //vetor de indices dos meus vizinhos
    double temp;       //armazena informacoes temporarias, como a distancia entre individuos
};

Individuo *populacao;

//prototipacao das funcoes
void inicializacao();
void cruzamento(int p, Individuo &filho1, Individuo &filho2);
void mutacao(Individuo &solucao);
void aptidao(Individuo &solucao);
void recombinacao(int p, Individuo &filho1, Individuo &filho2);
void calcularDTLZ2(double *x, double *fx);
void calcularDTLZ3(double *x, double *fx);
int lerArquivos(char *arquivo);
void calcularVizinhanca(int individuo);
double distanciaEuclidiana(double *x, double *y, int tamanho);
double TCH(double *sol, double *peso, double *z);

int main(const int argc, const char *argv[])
{
    srand(time(NULL)); //inicializa a semente aleatória com o tempo atual

    sprintf(pesos, "W_%dD", dimensoes_obj);

    tam_pop = lerArquivos(pesos);
    populacao = (Individuo *)malloc(tam_pop * sizeof(Individuo));

    for (int i = 0; i < tam_pop; i++) //atribui um peso a cada individuo
        memcpy(&populacao[i].peso, &dados[i], sizeof(double) * dimensoes_obj);

    for (int i = 0; i < dimensoes_obj; i++) //inicializa o vetor ideal com valores ruins para atualizar depois
        ideal[i] = DBL_MAX;

    inicializacao(); //inicializa o algoritmo, definindo valores iniciais para as soluções, pode ser aleatório

    for (int p = 0; p < tam_pop; p++)
        aptidao(populacao[p]); //calculo da aptidao, fitness ou valor objetivo

    for (long g = 0; g < geracoes; g++)
    { //laco principal
        for (int p = 0; p < tam_pop; p++)
        {
            Individuo filho1, filho2;
            cruzamento(p, filho1, filho2); //aqui esta implementado o cruzamento com a etapa de selecao dos pais junto.

            mutacao(filho1); //mutacao, acontece em cada individuo com uma dada probabilidade
            mutacao(filho2); //mutacao, acontece em cada individuo com uma dada probabilidade

            aptidao(filho1);                 //calculo da aptidao, fitness ou valor objetivo
            aptidao(filho2);                 //calculo da aptidao, fitness ou valor objetivo
            recombinacao(p, filho1, filho2); // compara os filhos com os vizinhos e subsitui se for o caso
        }
    }

    for (int i = 0; i < tam_pop; i++)
    {
        for (int j = 0; j < dimensoes_obj; j++)
            printf("%.3f ", populacao[i].fx[j]);
        printf(" -- ");
        for (int j = 0; j < dimensoes_var; j++)
        {
            printf("%.3f ", populacao[i].x[j]);
        }
        printf("\n");
    }
            // for(int i=0;i<tam_pop;i++){
            //     for(int j=0;j<T;j++){
            //         printf("%d ", populacao[i].vizinhanca[j]);
            //     }
            //     printf("\n");
            // }

    free(populacao);
}


void inicializacao()
{
    for (int i = 0; i < tam_pop; i++)
    {
        for (int j = 0; j < dimensoes_var; j++)
        {
            populacao[i].x[j] = rand() / (double)RAND_MAX;
        }
        calcularVizinhanca(i);
    }
}


void cruzamento(int p, Individuo &filho1, Individuo &filho2)
{
    Individuo pai1, pai2;

    //selecao dos pais -- a propria solucao e uma vizinha
    memcpy(&pai1, &populacao[p], sizeof(Individuo));
    memcpy(&pai2, &populacao[populacao[p].vizinhanca[rand() % T]], sizeof(Individuo));

    //CRUZAMENTO SBX
    double y1, y2, betaq;
    double distributionIndex = 30.0; //Parametro da distribuicao de valores
    for (int i = 0; i < dimensoes_var; i++)
    {
        double x1 = pai1.x[i];
        double x2 = pai2.x[i];
        if (rand() / (double)RAND_MAX <= 0.5)
        {
            if (abs(x1 - x2) > 1.0e-14)
            { //menor diferenca permitida entre valores
                if (x1 < x2)
                {
                    y1 = x1;
                    y2 = x2;
                }
                else
                {
                    y1 = x2;
                    y2 = x1;
                }
                double rnd = rand() / (double)RAND_MAX;
                double beta = 1.0 + (2.0 * (y1 - limiteInferior) / (y2 - y1));
                double alpha = 2.0 - pow(beta, -(distributionIndex + 1.0));

                if (rnd <= (1.0 / alpha))
                {
                    betaq = pow(rnd * alpha, (1.0 / (distributionIndex + 1.0)));
                }
                else
                {
                    betaq = pow(1.0 / (2.0 - rnd * alpha), 1.0 / (distributionIndex + 1.0));
                }
                double c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1));

                beta = 1.0 + (2.0 * (limiteSuperior - y2) / (y2 - y1));
                alpha = 2.0 - pow(beta, -(distributionIndex + 1.0));

                if (rnd <= (1.0 / alpha))
                {
                    betaq = pow((rnd * alpha), (1.0 / (distributionIndex + 1.0)));
                }
                else
                {
                    betaq = pow(1.0 / (2.0 - rnd * alpha), 1.0 / (distributionIndex + 1.0));
                }
                double c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1));

                if (c1 < limiteInferior)
                    c1 = limiteInferior;

                if (c2 < limiteInferior)
                    c2 = limiteInferior;

                if (c1 > limiteSuperior)
                    c1 = limiteSuperior;

                if (c2 > limiteSuperior)
                    c2 = limiteSuperior;

                if (rand() / (double)RAND_MAX <= 0.5)
                {
                    filho1.x[i] = c2;
                    filho2.x[i] = c1;
                }
                else
                {
                    filho1.x[i] = c1;
                    filho2.x[i] = c2;
                }
            }
            else
            {
                filho1.x[i] = x1;
                filho2.x[i] = x2;
            }
        }
        else
        {
            filho1.x[i] = x2;
            filho2.x[i] = x1;
        }
    }
}


void mutacao(Individuo &solucao)
{
    double probabilidade = 1.0 / dimensoes_var;
    double distributionIndex = 30.0;
    double rnd, delta1, delta2, mut_pow, deltaq;
    double y, yl, yu, val, xy;

    if ((rand() / (double)RAND_MAX) < prob_mutacao)
    {
        //Mutacao uniforme
        for (int var = 0; var < dimensoes_var; var++)
        {
            if ((rand() / (double)RAND_MAX) <= probabilidade)
            {
                y = solucao.x[var];
                yl = limiteInferior;
                yu = limiteSuperior;
                delta1 = (y - yl) / (yu - yl);
                delta2 = (yu - y) / (yu - yl);
                rnd = (rand() / (double)RAND_MAX);
                mut_pow = 1.0 / (distributionIndex + 1.0);
                if (rnd <= 0.5)
                {
                    xy = 1.0 - delta1;
                    val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (pow(xy, (distributionIndex + 1.0)));
                    deltaq = pow(val, mut_pow) - 1.0;
                }
                else
                {
                    xy = 1.0 - delta2;
                    val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (pow(xy, (distributionIndex + 1.0)));
                    deltaq = 1.0 - (pow(val, mut_pow));
                }
                y = y + deltaq * (yu - yl);
                if (y < yl)
                    y = yl;
                if (y > yu)
                    y = yu;
                solucao.x[var] = y;
            }
        }
    }
}


void aptidao(Individuo &solucao)
{
    calcularDTLZ2(solucao.x, solucao.fx);
    //calcularDTLZ3(solucao.x, solucao.fx);
    //atualizacao do vetor ideal
    for (int i = 0; i < dimensoes_obj; i++)
    { //atualizacao da solucao ideal
        if (solucao.fx[i] < ideal[i])
        {
            ideal[i] = solucao.fx[i];
        }
    }
}


void recombinacao(int p, Individuo &filho1, Individuo &filho2)
{
    //passo 2.5 do pseudocodigo
    //lembrando que uma solução é vizinha de si mesma
    //laco para percorrer a vizinhanca de um dado individuo
    for (int a=0; a<T; a++)
    {
        //considerando que os index dos vinhos foram amazenados no vetor de inteiros "vizinhanca"
        //recupera o indice "A" do vizinho de "P"
        int index_vizinho = populacao[p].vizinhanca[a];
        
        //calcula o THC do vizinho "A" de "P"         || considera o vetor de peso dele
        double thc_vizinho = TCH(populacao[a].fx, populacao[index_vizinho].peso, ideal);
        
        //calcula o THC do filho 1         || considera o vetor de peso do vizinho "A" de "P"
        double thc_filho1 = TCH(filho1.fx, populacao[index_vizinho].peso, ideal);

        if (thc_vizinho <= thc_filho1)
        {
            //atualiza os valores das variaveis de decisao - X
            for (int b=0; b<dimensoes_var; b++)
            {
                populacao[a].x[b] = filho1.x[b];
            }
            //atualiza os valores objetivos - F(x)
            for (int b=0; b<dimensoes_obj; b++)
            {
                populacao[a].fx[b] = filho1.fx[b];
            }
        }
        
        //calcula o THC do filho 2         || considera o vetor de peso do vizinho "A" de "P"
        double thc_filho2 = TCH(filho2.fx, populacao[index_vizinho].peso, ideal);
        if (thc_vizinho <= thc_filho1)
        {
            //atualiza os valores das variaveis de decisao - X
            for (int b=0; b<dimensoes_var; b++)
            {
                populacao[a].x[b] = filho2.x[b];
            }
            //atualiza os valores objetivos - F(x)
            for (int b=0; b<dimensoes_obj; b++)
            {
                populacao[a].fx[b] = filho2.fx[b];
            }
        }
    }
}



void calcularDTLZ2(double *x, double *fx)
{
    int k = dimensoes_var - dimensoes_obj + 1;
    double g = 0.0;

    for (int i = dimensoes_var - k; i < dimensoes_var; i++)
        g += (x[i] - 0.5) * (x[i] - 0.5);

    for (int i = 0; i < dimensoes_obj; i++)
        fx[i] = (1.0 + g);

    for (int i = 0; i < dimensoes_obj; i++)
    {
        for (int j = 0; j < dimensoes_obj - (i + 1); j++)
            fx[i] *= cos(x[j] * 0.5 * M_PI);
        if (i != 0)
        {
            int aux = dimensoes_obj - (i + 1);
            fx[i] *= sin(x[aux] * 0.5 * M_PI);
        }
    }
}


void calcularDTLZ3(double *x, double *fx)
{
    int k = dimensoes_var - dimensoes_obj + 1;
    double g = 0.0;

    for (int i = dimensoes_var - k; i < dimensoes_var; i++)
        g += (x[i] - 0.5) * (x[i] - 0.5) - cos(20.0 * M_PI * (x[i] - 0.5));

    g = 100.0 * (k + g);
    for (int i = 0; i < dimensoes_obj; i++)
        fx[i] = 1.0 + g;

    for (int i = 0; i < dimensoes_obj; i++)
    {
        for (int j = 0; j < dimensoes_obj - (i + 1); j++)
            fx[i] *= cos(x[j] * 0.5 * M_PI);
        if (i != 0)
        {
            int aux = dimensoes_obj - (i + 1);
            fx[i] *= sin(x[aux] * 0.5 * M_PI);
        }
    }
}

//funcao usada para a leitura de arquivos
void parse(char *record, char *delim, char arr[][1024], int *fldcnt)
{
    char *p = strtok(record, delim);
    int fld = 0;

    while (p != NULL)
    {
        strcpy(arr[fld], p);
        fld++;
        p = strtok(NULL, delim);
    }
    *fldcnt = fld;
}

//funcao que le um arquivo e joga seus valores em uma matriz na memoria
int lerArquivos(char *arquivo)
{
    char tmp[4096];
    int fldcnt = 0;
    char arr[1000][1024];
    int recordcnt = 0;
    FILE *in = fopen(arquivo, "r"); // abrir o arquivo

    if (in == NULL)
    {
        perror("Error opening the file");
        exit(EXIT_FAILURE);
    }
    while (fgets(tmp, sizeof(tmp), in) != 0) //le uma entrada
    {
        if (tmp[0] != '#' && tmp[0] != '\n')
        {
            parse(tmp, (char *)" \t", arr, &fldcnt); // quebra a entrada em campos
            //if(fldcnt>1)
            //	dimensoes_obj=fldcnt;
            for (int coluna = 0; coluna < fldcnt; coluna++)
            {
                for (int i = 0; i < strlen(arr[coluna]); i++)
                {
                    if (arr[coluna][i] == ',')
                        arr[coluna][i] = '.';
                }
                dados[recordcnt][coluna] = (double)atof(arr[coluna]);
            }
            recordcnt++;
        }
    }
    fclose(in);
    return recordcnt;
}

/*
    para cada individuo, seus vizinhos são os T vetores de peso mais próximos de acordo com a distancia euclidiana
*/
void calcularVizinhanca(int individuo)
{
    double sum;
    double maior_valor=0;
    int indice_maior;
    //replicacao do vetor de vizinhaca amazenando, no entanto, o indice e a distancia
    double distancias_armazenadas[T];
    //variavel de controle de insercao
    int contador = 0;
    //o individuo eh vizinho dele mesmo
    populacao[individuo].vizinhanca[0] = individuo;
    distancias_armazenadas[0]=0;
    contador++;
    //percorre a populacao
    for (int a = 0; a < tam_pop; a++)
    {
        sum = distanciaEuclidiana(populacao[individuo].peso, populacao[a].peso, dimensoes_obj);

        // cout << "distanciaEuclidiana == " << sum << endl;
        //enquanto o vetor de vizinhanca nao estiver totalmente cheio, ele ira inserir todo os individuos
        if (contador < T)
        {
            populacao[individuo].vizinhanca[contador] = a;
            distancias_armazenadas[contador]=sum;
            contador++;
            // cout << "{ CONTADOR < T } | contado==" << contador << " ||  [ INDICE ADD == " << a <<" ]" << endl;
        }
        
        //se o vetor ja estiver preenchido, sobrescrever o de maior distancia
        else
        {  
            //percorre toda vizinhanca
            for (int b=1; b<T; b++)
            {
                //compara o valor atual de "maior_valor" com "distancias_armazenadas[b]"
                // cout << " --- contador da vizinhanca: " << b << endl;
                if (distancias_armazenadas[b] > maior_valor)
                {
                    // cout << "( distancias_armazenadas[b](" << distancias_armazenadas[b] << ") > maior_valor(" << maior_valor << ") )" << endl;
                    //se for maior substitui
                    maior_valor=distancias_armazenadas[b];
                    //armazena o indice do maior
                    indice_maior=b;
                }
            }
            //sobrescreve o vizinho de maior distancia
            populacao[individuo].vizinhanca[indice_maior]=a;
            //atualiza repositorio local de valores das distancia
            distancias_armazenadas[indice_maior]=sum;
            // cout << " ||  [ INDICE ADD == " << a << " ( " << sum << " ) ]" << endl;
        }
        // cout << maior_valor << endl;
    }
}


double distanciaEuclidiana(double *x, double *y, int tamanho)
{ //calcula a distancia Euclidiana
    double sum = 0;
    for (int i = 0; i < tamanho; i++)
    {
        sum += ((x[i] - y[i]) * (x[i] - y[i]));
    }
    return sqrt(sum);
}


double TCH(double *sol, double *peso, double *z)
{ //Calcula a escalarizacao tchebycheff
    double maior = -DBL_MAX;
    for (int i = 0; i < dimensoes_obj; i++)
    {
        double valor = peso[i] * abs(sol[i] - z[i]);
        if (valor > maior)
            maior = valor;
    }
    return maior;
}
