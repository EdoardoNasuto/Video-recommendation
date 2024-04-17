def recommend_top_videos(user_index, R, nR, titles, themes, n=3):
    """
    Recommande les meilleures vidéos pour un utilisateur donné en se basant sur les notations prédites.

    Args:
        user_index (int): L'index de l'utilisateur pour lequel les recommandations sont générées.
        R (numpy.ndarray): La matrice de notation originale.
        nR (numpy.ndarray): La matrice de notations prédites.
        titles (list): Une liste contenant les titres des vidéos.
        themes (list): Une liste contenant les thèmes des vidéos.
        n (int, optional): Le nombre de vidéos à recommander (par défaut est 3).

    Returns:
        list: Une liste de tuples contenant les informations des vidéos recommandées. Chaque tuple contient
              le titre de la vidéo, son thème et la notation prédite par l'utilisateur.

    Note:
        Cette fonction utilise la matrice de notations prédites (nR) pour recommander les meilleures vidéos
        pour un utilisateur donné. Elle filtre les vidéos que l'utilisateur n'a pas encore regardées, puis
        classe les vidéos restantes par ordre décroissant de leurs notations prédites, et enfin sélectionne
        les n meilleures vidéos pour les recommander.
    """

    # Obtenir les notations prédites pour l'utilisateur donné
    notations_utilisateur = nR[user_index]

    # Trouver les indices des vidéos que l'utilisateur n'a pas encore regardées (notation égale à 0)
    indices_non_regardes = []
    for i, notation in enumerate(R[user_index]):
        if notation == 0:
            indices_non_regardes.append(i)

    # Filtrer les prédictions pour ne considérer que les vidéos non regardées
    predictions_non_regardes = []
    for index, prediction in enumerate(notations_utilisateur):
        if index in indices_non_regardes:
            predictions_non_regardes.append((index, prediction))

    # Trier les indices par ordre décroissant des prédictions
    indices_tries = sorted(predictions_non_regardes,
                           key=lambda x: x[1], reverse=True)

    # Sélectionner les n premiers indices (vidéos les mieux notées)
    indices_top = []
    for index, _ in indices_tries[:n]:
        indices_top.append(index)

    # Générer les recommandations avec les informations nécessaires
    recommandations = []
    for i in indices_top:
        recommandations.append(
            (titles[i], themes[i], notations_utilisateur[i]))

    return recommandations
