import csv


def load_data(data):
    """
    Charge les données à partir d'un fichier CSV.

    Args:
        data (str): Le chemin vers le fichier CSV contenant les données.

    Returns:
        tuple: Un tuple contenant la matrice de notation R, la liste des thèmes et la liste des titres.

    Note:
        Cette fonction lit un fichier CSV contenant les données sous la forme suivante :
        - La première ligne contient les thèmes des vidéos.
        - La deuxième ligne contient les titres des vidéos.
        - Les lignes suivantes contiennent les notations des utilisateurs pour chaque vidéo,
          avec chaque ligne représentant un utilisateur et chaque colonne représentant une vidéo.
        Les notations doivent être des entiers.
    """
    # Initialisation de la matrice de notation R
    R = []

    # Ouverture du fichier CSV en mode lecture
    with open(data, 'r') as file:
        # Création d'un objet reader pour lire le fichier CSV avec le délimiteur ';'
        reader = csv.reader(file, delimiter=';')

        # Lecture de la première ligne du fichier (thèmes des vidéos)
        themes = next(reader)

        # Lecture de la deuxième ligne du fichier (titres des vidéos)
        titles = next(reader)

        # Parcours des lignes restantes dans le fichier (notations des utilisateurs)
        for row in reader:
            # Initialisation d'une liste pour stocker les notations d'un utilisateur
            ligne = []

            # Parcours des colonnes dans la ligne actuelle
            for column in row:
                # Conversion de la notation en entier et ajout à la liste
                ligne.append(int(column))

            # Ajout de la liste de notations de l'utilisateur à la matrice de notation R
            R.append(ligne)

    # Retourne la matrice de notation R, la liste des thèmes et la liste des titres
    return R, themes, titles
