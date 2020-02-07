import numpy as np

def combine_results(type1,type2,type3,type4,weight1,weight2,weight3,weight4):
    result_files=["result5","result10","result20"]

    for file in result_files:
        #Creating output file
        resultfile = open(file+"combined.txt", "a")

        #Reading predictions from result file of Cosine Amplification
        cosine_results = open(file+type1+".txt", 'r').read().strip().split('\n')
        cosine_matrix = [data.split() for data in cosine_results]
        cosine_matrix = [[int(e) for e in data] for data in cosine_matrix]

        # Reading predictions from result file of Pearson with IUF
        pearson_results = open(file+type2+".txt", 'r').read().strip().split('\n')
        pearson_matrix = [data.split() for data in pearson_results]
        pearson_matrix = [[int(e) for e in data] for data in pearson_matrix]

        # Reading predictions from result file of Cosine with 50 neighbors
        cosine_raw_results = open(file + type3 + ".txt", 'r').read().strip().split('\n')
        cosine_raw_matrix = [data.split() for data in cosine_raw_results]
        cosine_raw_matrix = [[int(e) for e in data] for data in cosine_raw_matrix]

        # Reading predictions from result file of Cosine with IUF
        cosine_cuf_results = open(file + type4 + ".txt", 'r').read().strip().split('\n')
        cosine_cuf_matrix = [data.split() for data in cosine_cuf_results]
        cosine_cuf_matrix = [[int(e) for e in data] for data in cosine_cuf_matrix]


        for i in range(len(cosine_matrix)):
            user_id = cosine_matrix[i][0]
            movie_id = cosine_matrix[i][1]
            rating_Cosine = cosine_matrix[i][2]
            rating_Pearson = pearson_matrix[i][2]
            rating_raw_Cosine = cosine_raw_matrix[i][2]
            rating_cuf_cosine = cosine_cuf_matrix[i][2]

            #Calculating weghted average depending on weights assigned to each algorithm
            weighted_avg = (weight1*rating_Cosine)
            weighted_avg+= (weight2*rating_Pearson)
            weighted_avg+= (weight3*rating_raw_Cosine)
            weighted_avg+= (weight4*rating_cuf_cosine)

            weighted_avg= weighted_avg/(weight1+weight2+weight3+weight4)

            #Rouding to nearest value
            rating= int(np.rint(weighted_avg))

            #Appending to output file
            resultfile.write("%i %i %i\n" % (user_id, movie_id, rating))
    resultfile.close





def main():
    #Combining results from Cosine with case amplification,pearson with IUF,Cosine raw,cosine with IUF
    combine_results("C","P","CR","CUF",2,5,1,1)
main()