import numpy
from sklearn.cluster import KMeans


def matrix_initialization(R, N, M, K):
    """
    Initialise les matrices P et Q en utilisant la méthode de clustering K-means.

    Args:
        R (numpy.ndarray): La matrice de notation.
        N (int): Le nombre d'utilisateurs.
        M (int): Le nombre de vidéos.
        K (int): Le nombre de clusters à utiliser.

    Returns:
        tuple: Un tuple contenant les matrices P et Q initialisées.

    Note:
        Cette fonction initialise les matrices P et Q en utilisant la méthode de clustering K-means.
        Elle clusterise d'abord les utilisateurs en K groupes, puis initialise la matrice P.
        Ensuite, elle clusterise les vidéos en K groupes, puis initialise la matrice Q.
        Cela permet d'initialiser les matrices P et Q de manière à regrouper
        les utilisateurs et les vidéos avec des caractéristiques similaires.
    """

    # Clustering des utilisateurs avec K-means
    user_clusters = KMeans(n_clusters=K).fit_predict(R)

    # Initialisation de la matrice P à partir des clusters des utilisateurs
    P = numpy.array([numpy.roll(user_clusters/K, i*int(N/K))
                    for i in range(K)]).T

    # Clustering des vidéos avec K-means
    video_clusters = KMeans(n_clusters=K).fit_predict(R.T)

    # Initialisation de la matrice Q à partir des clusters des vidéos
    Q = numpy.array([numpy.roll(video_clusters/K, i*int(M/K))
                    for i in range(K)]).T

    return P, Q
