import numpy as np

def feed_training_data(input):
    i=0
    file = open('train.txt', 'r')
    for line in file.readlines():
        input[i] = [int(x) for x in line.split()]
        i=i+1

def test_cosine_similarity(input,file,num,k):
    testdata = open(file, 'r').read().strip().split('\n')
    testmatrix = [data.split() for data in testdata]
    testmatrix = [[int(e) for e in data] for data in testmatrix]
    current_user_id = testmatrix[0][0]-1
    current_user_data = [0]*1000
    movie_ids = []
    ratings=[]
    outputfile = "result"+str(num)+".txt"

    for i in range(len(testmatrix)):
        user_id = testmatrix[i][0]-1
        movie_id = testmatrix[i][1]-1
        rating = testmatrix[i][2]
        print(movie_id)
        if user_id== current_user_id:
            if rating == 0 :
                movie_ids.append(movie_id)
            else:
                current_user_data [movie_id] = rating
        else:
            get_results_using_cosine(
                input,
                current_user_data,
                current_user_id,
                movie_ids,
                ratings,
                outputfile,
                k,
                None,
                None
            )
            current_user_id = user_id
            current_user_data = [0] * 1000
            movie_ids = []
            if rating == 0 :
                movie_ids.append(movie_id)
            else:
                current_user_data [movie_id] = rating

    get_results_using_cosine(input,current_user_data,current_user_id,movie_ids,ratings,outputfile,k,None,None)
    return ratings

def get_results_using_cosine(input,current_user_data,current_user_id,movie_ids,ratings,outputfile,k,caseAmp,isIUF):
     weights=[]
     current_users_avg = [np.average([r for r in current_user_data if r > 0])]
     IUFratings=np.copy(input)

     # If IUF multiplying input matrix with IUF factor
     if isIUF is not None:
         for i in range(1000):
             count=0
             for row in input:
                if row[i] > 0:
                    count=count+1
             if count == 0:
                 continue
             iuf=np.log(200/count)
             current_user_data[i] = np.rint(current_user_data[i] *iuf)
             for u in IUFratings:
                 u[i] = np.rint(u[i] * iuf)


     if isIUF is not None:
         for row in IUFratings:
             weights.append(get_cosine_similarity(row, current_user_data))
     else:
        for row in input:
            weights.append(get_cosine_similarity(row,current_user_data))

     # If case amplification calculating new weights
     if caseAmp is not None:
         weights = [w * np.abs(w)**(caseAmp-1) for w in weights]

     # Creating output file
     file = open(outputfile, "a")

     for movie in movie_ids:
         sum=0
         weight_sum=0
         rating =0
         results = []
         for i, val in enumerate(weights):
             if input[i][movie] == 0:
                 continue
             results.append((i, val))

         #Finding k nearest neighbors

         results.sort(key=sortSecond, reverse=True)
         del results[k:]

         #Predicting rating based on weighted average
         for row in results:
             rating=0
             user_id = row[0]
             sim = row[1]

             #If user has not rated the movie ignore the user
             if input[user_id][movie]==0:
                 continue

             weight_sum+=sim
             sum+=sim*input[user_id][movie]

         if weight_sum != 0:
             rating = sum/weight_sum
         else:
             # If no neighbors found take average rating of current user
             rating = current_users_avg

         rating = int(np.rint(rating))

         # Appending results to output file
         file.write("%i %i %i\n" % (current_user_id+1, movie+1,rating))
         ratings.append(rating)

     file.close()
     return

def get_common_values(v1, v2):
    """
    Returns new vectors (a, b) where both vectors have values for given index
    """
    v1_new = []
    v2_new = []
    for i, x in enumerate(v1):
        y = v2[i]
        if y > 0 and x > 0:
            v1_new.append(x)
            v2_new.append(y)
    return np.array(v1_new), np.array(v2_new)

def get_cosine_similarity(user1,user2):
    """
        Cosine similarity between two vectors.
        Value ranges from [-1, 1].
    """

    vector_v1, vector_v2 = get_common_values(user1, user2)

    sim = np.dot(vector_v1, vector_v2)

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
    N = 2000
    # Declaring columns
    M = 1000

    #initializing input matrix
    input = [[0 for i in range(M)] for j in range(N)]
    feed_training_data(input)

    print("processing test file 5")
    test_cosine_similarity(input,"test5.txt",5,50)
    print("predictions done for file 5")
    print("processing test file 10")
    test_cosine_similarity(input,"test10.txt",10,50)
    print("predictions done for file 10")
    print("processing test file 20")
    test_cosine_similarity(input, "test20.txt",20,50)
    print("predictions done for file 20")

main()
