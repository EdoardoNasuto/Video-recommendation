import numpy


def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    """
    Effectue la factorisation de la matrice R en utilisant la méthode de factorisation matricielle.

    Args:
        R (numpy.ndarray): La matrice de notation.
        P (numpy.ndarray): La matrice des caractéristiques des utilisateurs (taille |U| * K).
        Q (numpy.ndarray): La matrice des caractéristiques des vidéos (taille |D| * K).
        K (int): Le nombre de caractéristiques latentes.
        steps (int, optional): Le nombre d'itérations (par défaut est 5000).
        alpha (float, optional): Le taux d'apprentissage qui permet de ne pas surestimer les données observées (par défaut est 0.0002).
        beta (float, optional): Le paramètre de régularisation (par défaut est 0.02).

    Returns:
        tuple: Un tuple contenant les matrices des caractéristiques des utilisateurs et des vidéos.

    Note:
        Cette fonction utilise la méthode de factorisation matricielle pour approximer la matrice de notation R
        en minimisant l'erreur quadratique entre les valeurs réelles de R et les prédictions basées sur les matrices P et Q.
    """

    # Transposition de la matrice Q pour faciliter les calculs
    Q = Q.T

    # Boucle d'apprentissage
    for _ in range(steps):
        # Calcul de l'erreur totale après chaque itération
        e = 0

        # Parcours de chaque utilisateur
        for i in range(len(R)):
            # Parcours de chaque vidéo
            for j in range(len(R[i])):
                # Si la notation existe
                if R[i][j] > 0:
                    # Calcul de la prédiction
                    pred = numpy.dot(P[i], Q[:, j])
                    # Calcul de l'erreur
                    eij = R[i][j] - pred
                    e += eij ** 2
                    # Régularisation de l'erreur
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))

                    # Mise à jour des caractéristiques des utilisateurs et des vidéos
                    for k in range(K):
                        P[i][k] += alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] += alpha * (2 * eij * P[i][k] - beta * Q[k][j])

        # Si l'erreur est inférieure à un seuil, arrêtez l'entraînement
        if e < 0.001:
            break

    # Retourne les matrices des caractéristiques des utilisateurs et des vidéos
    return P, Q.T
