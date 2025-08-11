import heapq
import matplotlib.pyplot as plt

b = [200, 200, 300, 700, 1000, 200]
a = 2000
s= 2
class GrapheDictionnaire:
    def __init__(self):
        self.graphe = {}
        self.visiter = {}  # Dictionnaire pour suivre l'état de visite des sommets
        self.prix = {}

    def ajouter_sommet(self, s,pr):
        if s not in self.graphe:
            self.graphe[s] = {}
            self.visiter[s] = False  # Initialise l'état de visite du sommet
            self.prix[s] = pr #Initialiser le prix de l'installation.

    def ajouter_arete(self, s1, s2, p):
        if s1 in self.graphe and s2 in self.graphe:
            self.graphe[s1][s2] = p

    def supprimer_sommet(self, s):
        if s in self.graphe:
            del self.graphe[s]
            del self.visiter[s]  # Supprime également son état de visite
            for sommet in self.graphe:
                self.graphe[sommet].pop(s, None)  # Supprime l'arête sans erreur si elle n'existe pas

    def supprimer_arete(self, s1, s2):
        if s1 in self.graphe:
            self.graphe[s1].pop(s2, None)  # Supprime l'arête

    def visiter(self, s):
        """Marque un sommet comme visité."""
        if s in self.visiter:
            self.visiter[s] = True

    def est_visite(self, s):
        """Retourne True si le sommet a été visité, sinon False."""
        return self.visiter.get(s, False)

    def afficher_graphe(self):
        """Affiche la structure du graphe."""
        for sommet, voisins in self.graphe.items():
            print(f"{sommet} -> {voisins}")
    
    def contient_cycle(self):
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
        
            visited.add(node)
            rec_stack.add(node)
            for neighbor in self.graphe.get(node, {}):
                print("Pile:",rec_stack)
                if dfs(neighbor):  # Si un cycle est détecté, on sort immédiatement
                    return True
            
            rec_stack.remove(node)  # On enlève le noeud après la boucle
            return False
        
        return any(dfs(node) for node in self.graphe)
    


def dijkstra(graphe, depart):
    """
    Algorithme de Dijkstra
    Retourne les coûts min depuis `depart` et les prédécesseurs pour reconstruction du chemin.
    """
    sommets = list(graphe.graphe.keys())
    cout = {sommet: float('inf') for sommet in sommets}
    precedent = {sommet: None for sommet in sommets}
    cout[depart] = 0

    # File de priorité : (cout, sommet)
    file = [(0, depart)]  

    while file:
        d, u = heapq.heappop(file) 
        if d > cout[u]:
            continue  

        for v, nombre in graphe.graphe[u].items():
            alt = cout[u] + nombre 
            if alt < cout[v]: 
            #Si le cout calculé est inférieur au cout défini dans graphe, on le garde.
                cout[v] = alt
                precedent[v] = u #Mise à jour du précédent
                heapq.heappush(file, (alt, v)) 
                #On l'ajoute dans la file pour explorer ses autres voisins

    return cout, precedent

def chemin_dijkstra(precedent, cible):
    chemin = []
    while cible is not None:
        chemin.insert(0, cible)
        cible = precedent[cible]
    return chemin

# Politique directeur achats
def politique_directeur_achats(b, a, s):
    stock = 0
    for i in range(len(b)):
        stock += b[i]*i
    return a + stock*s
def politique_directeur_achatsl(b, a, s):
    stock = 2000
    valeurs_directeura = []
    for i in range(len(b)):
        stock += b[i]*i*s
        valeurs_directeura.append(stock)
    return valeurs_directeura
# Politique directeur financier
def politique_directeur_financier(n, a):
    return n * a


# Création du graphe
graphe = GrapheDictionnaire()
graphe.ajouter_sommet(0,200)
graphe.ajouter_sommet(1,200)
graphe.ajouter_sommet(2,200)
graphe.ajouter_sommet(3,300)
graphe.ajouter_sommet(4,700)  
graphe.ajouter_sommet(5,1000)  
graphe.ajouter_sommet(6,200)
for i in range(0,6): #Création des arêtes 2000
    graphe.ajouter_arete(i, i+1, 2000)
graphe.ajouter_arete(0,2,2400)
graphe.ajouter_arete(0,3,3600)
graphe.ajouter_arete(1,3,2600)
graphe.ajouter_arete(2,4,3400)
graphe.ajouter_arete(3,5,4000)
graphe.ajouter_arete(3,6,4800)
graphe.ajouter_arete(4, 6,a+2*(+graphe.prix[6]))

cout, precedent = dijkstra(graphe, 0)
print(cout,precedent)
chemin = chemin_dijkstra(precedent, 6)
print("Coût minimal de 0 à 6 avec Dijkstra :", cout[6], "€")
print("Chemin trouvé par Dijkstra :", chemin)
dist, precedent = dijkstra(graphe, 0)
chemin = chemin_dijkstra(precedent, 6)



# Comparaison

print("graphe avec cycle")
graphe.afficher_graphe() #Cas Graphe cyclique

g = GrapheDictionnaire()
g.ajouter_sommet(1,200)
g.ajouter_sommet(2,200)
g.ajouter_sommet(3,300)
g.ajouter_sommet(4,300)
g.ajouter_arete(1,2,1000)
g.ajouter_arete(2,3,1000)
g.ajouter_arete(3,1,1000)
g.ajouter_arete(2,4,1000)
g.afficher_graphe() #Cas graphe acyclique
print("Le graphe contient un cycle:",graphe.contient_cycle())
print("Le graphe contient un cycle:",g.contient_cycle())

print("\n--- Comparaison des politiques ---")
print("Directeur des achats     :", politique_directeur_achats(b, a, s), "€")
print(" Cout Directeur des achats     :", politique_directeur_achatsl(b, a, s), "€")

print("Directeur financier       :", politique_directeur_financier(len(b), a), "€")
print("Coûts opti:",dist)
print("Directeur général (opt.) :", dist[6], "€")
print("Chemin optimal :", chemin)





valeurs_financier_mois = []
cumul = 0
for i in range(6):  # Mois 0 à 6
    cumul += 2000
    valeurs_financier_mois.append(cumul)
print(valeurs_financier_mois) #Liste couts directeur financier
# Affichage humain du chemin
mois = [f"mois {i}" for i in range(7)]
for i in range(len(chemin)-1):
    print(f" - Commander en {mois[chemin[i]]} pour couvrir jusqu'à {mois[chemin[i+1]]}")

# Utilisation de dist pour tracer les coûts cumulés mois par mois de la politique optimale (Dijkstra)
mois_labels = [f"Mois {i}" for i in range(len(dist))]
valeurs_cumulees = [dist[i] for i in range(len(dist))]

# Tracé du graphique cumulé
plt.figure(figsize=(8,6))
plt.plot(mois_labels, valeurs_cumulees, marker='o', color='blue', linewidth=2, label="Directeur général (opt.)")
#plt.plot(mois_labels, valeurs_financier_mois, linestyle=':', marker='s', color='red', linewidth=2, label="Directeur financier (cumulé)")


# Ajouter les annotations de valeurs
for i, val in enumerate(valeurs_cumulees):
    plt.text(i, val + 150, f"{int(val)}€", ha='center', fontsize=10)


plt.title("Évolution du coût cumulé optimal mois par mois", fontsize=16)
plt.xlabel("Mois", fontsize=14)
plt.ylabel("Coût cumulé (€)", fontsize=14)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()


valeurs_financier_mois = []
cumul = 0
for i in range(6):  # Mois 0 à 6 => donc 6 valeurs
    cumul += 2000
    valeurs_financier_mois.append(cumul)

# Création des étiquettes de mois
mois_labels = [f"Mois {i+1}" for i in range(6)]  # Mois 1 à 6

# graphique pour le directeur financier
plt.figure(figsize=(8, 6))
plt.plot(mois_labels, valeurs_financier_mois, linestyle='-', marker='s', color='red', linewidth=2, label="Directeur financier (cumulé)")
plt.text(len(mois_labels)-0.5, 17800 + 150, "17800€", ha='right', fontsize=10, color='green')

# Ajouter les annotations de valeurs
for i, val in enumerate(valeurs_financier_mois):
    plt.text(i, val + 150, f"{int(val)}€", ha='center', fontsize=10, color='red')

plt.title("Évolution du coût cumulé - Directeur financier", fontsize=16)
plt.xlabel("Mois", fontsize=14)
plt.ylabel("Coût cumulé (€)", fontsize=14)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()


"""Graphique pour méthode directeur des achats"""
valeurs_achats_mois = politique_directeur_achatsl(b, a, s)

# Créer les étiquettes pour les mois correspondants
mois_labels = [f"Mois {i+1}" for i in range(len(valeurs_achats_mois))]  # Mois 1 à 5

# Tracé du graphique
plt.figure(figsize=(8, 6))
plt.plot(mois_labels, valeurs_achats_mois, linestyle='-', marker='o', color='green', linewidth=2, label="Directeur des achats")

# Ajouter les annotations de chaque valeur
for i, val in enumerate(valeurs_achats_mois):
    plt.text(i, val + 300, f"{val}€", ha='center', fontsize=10, color='green')

plt.title("Évolution du coût cumulé - Directeur des achats", fontsize=16)
plt.xlabel("Mois", fontsize=14)
plt.ylabel("Coût cumulé (€)", fontsize=14)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()





