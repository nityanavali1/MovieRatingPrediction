import numpy as np

def feed_training_data(input,filename):
    i=0
    file = open(filename, 'r')
    for line in file.readlines():
        input[i] = [int(x) for x in line.split()]
        i=i+1

def test_item_based_similarity(input,file,num):
    testdata = open(file, 'r').read().strip().split('\n')
    testmatrix = [data.split() for data in testdata]
    testmatrix = [[int(e) for e in data] for data in testmatrix]
    current_user_id = testmatrix[0][0]-1
    current_user_data={}
    movie_ids = []
    ratings=[]
    outputfile = "result"+str(num)+".txt"

    for i in range(len(testmatrix)):
        user_id = testmatrix[i][0]-1
        movie_id = testmatrix[i][1]-1
        rating = testmatrix[i][2]

        if user_id== current_user_id:
            if rating == 0 :
                movie_ids.append(movie_id)
            else:
                current_user_data [movie_id] = rating
        else:
            get_results_using_adjusted_cosine(
                input,
                current_user_data,
                current_user_id,
                movie_ids,
                ratings,
                outputfile
            )
            current_user_id = user_id
            current_user_data = {}
            movie_ids = []
            if rating == 0 :
                movie_ids.append(movie_id)
            else:
                current_user_data [movie_id] = rating

    get_results_using_adjusted_cosine(input,current_user_data,current_user_id,movie_ids,ratings,outputfile)
    return ratings

def get_results_using_adjusted_cosine(input,current_user_data,current_user_id,movie_ids,ratings,outputfile):

     #Creating movie-user matrix by taking transpose of inpur matrix
     movie__user_matrix = np.array(input).T

     #Calculating average rating for every user
     users_avg = [np.average([r for r in u if r > 0]) for u in input]

     #Creating output file
     file = open(outputfile, "a")

     for movie in movie_ids:
         sum=0
         weight_sum=0
         rating =0
         weights ={}

         for key in current_user_data:
             weights[key]=(get_cosine_similarity(movie__user_matrix[key], movie__user_matrix[movie],users_avg))


         for key_movie in current_user_data:

             weight_sum+=weights[key_movie]
             sum+=weights[key_movie]*current_user_data[key_movie]

         if weight_sum != 0:
             rating = sum/weight_sum
         else:
         # If no neighbors were found guessing rating as 3
             rating = 3

         rating = scale_rating(rating)

         #Appending to output file
         file.write("%i %i %i\n" % (current_user_id+1, movie+1,rating))
         ratings.append(rating)

     file.close()
     return

def scale_rating(rating):
    rating = int(np.rint(rating))
    if rating > 5:
        rating = 5
    elif rating < 1:
        rating = 1

    return rating
def get_common_values(v1, v2, users_avg):
    """
    Returns new vectors (a, b) after filtering any
    indices where an element in a or b <= 0.
    """
    v1_new = []
    v2_new = []
    for i, x in enumerate(v1):
        y = v2[i]
        if y > 0 and x > 0:
            v1_new.append(x-users_avg[i])
            v2_new.append(y-users_avg[i])
    return np.array(v1_new), np.array(v2_new)

def get_cosine_similarity(movie1,movie2,users_avg):
    """
        Cosine similarity between two vectors.
        Returns a float in [-1, 1].
    """

    vector_v1, vector_v2 = get_common_values(movie1, movie2, users_avg)

    sim = np.dot((vector_v1), vector_v2)

    lenght_v1 = get_length(vector_v1)
    length_v2 = get_length(vector_v2)

    if lenght_v1 != 0 and length_v2 != 0:
        sim /= (lenght_v1 * length_v2)
    else:
        sim = 0

    if sim > 1:
        sim = 1
    elif sim < -1:
        sim = -1

    return sim

def get_length(v):
    return np.sqrt(np.sum(np.square(v)))
def sortSecond(val):
    return val[1]
def main():
    # Declaring rows
    N = 200
    # Declaring columns
    M = 1000

    #Intializing input matrix
    input = [[0 for i in range(M)] for j in range(N)]
    feed_training_data(input,'train.txt')

    print("processing test file 5")
    test_item_based_similarity(input,"test5.txt",5)
    print("predictions done for file 5")
    print("processing test file 10")
    test_item_based_similarity(input,"test10.txt",10)
    print("predictions done for file 10")
    print("processing test file 20")
    test_item_based_similarity(input, "test20.txt",20)
    print("predictions done for file 20")

main()