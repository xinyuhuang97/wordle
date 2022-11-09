#Projet :wordle 2022/04
#Binome : Xinyu HUANG, Ruohui HU
#Numero : 3803966, 21102304
from constraint import *
import numpy as np
from copy import deepcopy

def readfile(file):
    """
    file->dict
     à partir le fichier dicto.txt, générer une dictionnaire dont les
     indices sont n(nombre de lettre) et les valeurs sont la liste des
     mots de ce longueur.
    """
    lines=[]
    dic_file=dict()
    with open(file) as f:
        for line in f:
            word=[i for i in line.rstrip()]
            lg_word=len(word)
            if lg_word in dic_file:
                dic_file[lg_word].append(word)
            else:
                dic_file[lg_word]=[word]
    return dic_file


def check_correct(instance, word):
    """
    str*str->int,int
    comparer la différence entre le mot généré par nous-même(instance)
    et le mot secret(mot) en retournant le nombre de lettres bien placés
    et mal placé.
    """
    bien_place=0
    mal_place=0
    list_not_match1=[]
    list_not_match2=[]
    for i in range(len(instance)):
        if instance[i]==word[i]:
            bien_place+=1
        else:
            list_not_match1.append(instance[i])
            list_not_match2.append(word[i])
    for letter in list_not_match1:
        if letter in list_not_match2:
            list_not_match2.remove(letter)
            mal_place+=1
    return bien_place, mal_place



def solver_a1(word, list_mot, list_domain,render=0):
    """
    word : mot secret
    list_mot : la dictionnaire(la liste de mots de même longueur que word)
    list_domaine : pour vairable xi(lettre positionné à la position i du mot),
                   généré son domaine depuis la dictionnaire.(Les lettres possibles pour xi)
    str * dict * list ->int
    """
    n=len(word)
    pb=Problem()
    #Fixer la domaine à l'aide de list_domain
    for i in range(n):
        name_variable=str(i)
        pb.addVariables(name_variable,list_domain[i])

    # Générer alératoirement un mot
    solutions=deepcopy(list_mot)
    bien_place=1
    index=np.random.randint(solutions.shape[0])
    solution=solutions[index]
    bien_place, mal_place = check_correct(solution, word )
    solutions=np.array([ solutions[i] for i in range(solutions.shape[0] ) if i!=index])
    counter=1
    true_count=0
    if render==0:
        print("Word to guess: ",word)
        print(counter, solution, word, bien_place, mal_place )

    #Quand bien_place < n
    while bien_place<n:
        counter+=1
        #Si bien_place==0 alors supprime de la domaine de x le valeur solution[i]
        #et supprime de la liste les mots qui ne satisfont plus la domaine
        if bien_place==0:
            for i in range(n):
                if solution[i] in pb._variables[str(i)]:
                    pb._variables[str(i)].remove(solution[i])
                    solutions=solutions[np.where(solutions[:,i]!=solution[i])]

        #Générer alératoirement un mot et compet le nombre de bien/mal placés
        index=np.random.randint(len(solutions))
        solution=solutions[index]
        solutions=np.array([solutions[i] for i in range(len(solutions) ) if i!=index])
        bien_place, mal_place = check_correct(solution, word)

        if render==0:
            print(counter, solution, word, bien_place, mal_place )
    return counter

def solver_a2(word, list_mot, list_domain,render=0):
    """
    word : mot secret
    list_mot : la dictionnaire(la liste de mots de même longueur que word)
    list_domaine : pour vairable xi(lettre positionné à la position i du mot),
                   généré son domaine depuis la dictionnaire.(Les lettres possibles pour xi)
    str * dict * list ->int
    """
    n=len(word)
    pb=Problem()

    #Fixer la domaine à l'aide de list_domain
    for i in range(n):
        name_variable=str(i)
        pb.addVariables(name_variable,list_domain[i])

    solutions=deepcopy(list_mot)
    bien_place=1

    # Générer alératoirement un mot
    index=np.random.randint(solutions.shape[0])
    solution=solutions[index]
    bien_place, mal_place = check_correct(solution, word )
    solutions=np.array([ solutions[i] for i in range(solutions.shape[0] ) if i!=index])
    counter=1
    if render==0:
        print("Word to guess: ",word)
        print(counter, solution, word, bien_place, mal_place )

    #Quand bien_place < n
    while bien_place<n:
        counter+=1
        #Si bien_place==0 alors supprime de la domaine de x le valeur solution[i]
        #et supprime de la liste les mots qui ne satisfont plus la domaine
        if bien_place==0:
            if mal_place==0:
                #(A2 Arc-consistant ) Si bien_place + mal_place==0 alors  supprime
                #de la domaine de xi tous les lettres apparaisent dans le mot solution
                #et supprime de la liste les mots qui ne satisfont plus la domaine
                for i in range(n):
                    for j in range(n):
                        if solution[i] in pb._variables[str(j)]:
                            pb._variables[str(j)].remove(solution[i])
                            solutions=solutions[np.where(solutions[:,j]!=solution[i])]
            else:
                for i in range(n):
                    if solution[i] in pb._variables[str(i)]:
                        pb._variables[str(i)].remove(solution[i])
                        solutions=solutions[np.where(solutions[:,i]!=solution[i])]

        #Générer alératoirement un mot et compet le nombre de bien/mal placés
        index=np.random.randint(len(solutions))
        solution=solutions[index]
        solutions=np.array([solutions[i] for i in range(len(solutions) ) if i!=index])
        bien_place, mal_place = check_correct(solution, word)

        if render==0:
            print(counter, solution, word, bien_place, mal_place )

    return counter


def generate_probabiliste_dict(list_mot, n):
    """
    list*int->dict
    qui génère une dictionnary qui retourne la valeur probabiliste
    pour un mot donné comme indice.
    """
    nb_dict=dict()
    pb_dict=dict()
    total=len(list_mot)
    for i in range(n):
        nb_dict[i]=dict()
    for mot in list_mot:
        for i in range(n):
            letter=mot[i]
            if letter in nb_dict[i]:
                nb=nb_dict[i][letter]
                nb_dict[i][letter]=nb+1
            else:
                nb_dict[i][letter]=1
    for index, mot in enumerate(list_mot):
        pb_dict[str(mot)]=np.sum([ np.log(nb_dict[i][mot[i]]/total) for i in range(n)])
    return pb_dict


def find_index(list_mot,mot):
    """
    list*str->int
    retourne l'indice pour un mot dans la liste
    """
    for index,m in enumerate(list_mot):
        if str(m)==str(mot):
            return index


def solver_csp_probabiliste(word, list_mot, list_domain, render=0):
    """
    word : mot secret
    list_mot : la dictionnaire(la liste de mots de même longueur que word)
    list_domaine : pour vairable xi(lettre positionné à la position i du mot),
                   généré son domaine depuis la dictionnaire.(Les lettres possibles pour xi)
    str * dict * list ->int
    """
    n=len(word)
    pb=Problem()
    #Fixer la domaine à l'aide de list_domain
    for i in range(n):
        name_variable=str(i)
        pb.addVariables(name_variable,list_domain[i])

    solutions=deepcopy(list_mot)
    bien_place=1

    # Générer un mot par max de valeur probabiliste
    pb_dict=generate_probabiliste_dict(list_mot, n)
    index=max(pb_dict, key=pb_dict.get)
    del pb_dict[index]
    index=find_index(solutions,index)
    solution=solutions[index]

    bien_place, mal_place = check_correct(solution, word )
    solutions=np.array([ solutions[i] for i in range(solutions.shape[0] ) if i!=index])
    counter=1
    if render==0:
        print("Word to guess: ",word)
        print(counter, solution, word, bien_place, mal_place )

    #Quand bien_place < n
    while bien_place<n:
        counter+=1
        change=0
        #Si bien_place==0 alors supprime de la domaine de x le valeur solution[i]
        #et supprime de la liste les mots qui ne satisfont plus la domaine
        if bien_place==0:
            change=1
            if mal_place==0:
                #(A2 Arc-consistant ) Si bien_place + mal_place==0 alors  supprime
                #de la domaine de xi tous les lettres apparaisent dans le mot solution
                #et supprime de la liste les mots qui ne satisfont plus la domaine
                for i in range(n):
                    for j in range(n):
                        if solution[i] in pb._variables[str(j)]:
                            pb._variables[str(j)].remove(solution[i])
                            solutions=solutions[np.where(solutions[:,j]!=solution[i])]
            else:
                for i in range(n):
                    if solution[i] in pb._variables[str(i)]:
                        pb._variables[str(i)].remove(solution[i])
                        solutions=solutions[np.where(solutions[:,i]!=solution[i])]
        if change==1:
            pb_dict=generate_probabiliste_dict(solutions, n)

        # Générer un mot par max de valeur probabiliste
        index=max(pb_dict, key=pb_dict.get)
        del pb_dict[index]
        index=find_index(solutions,index)
        solution=solutions[index]
        solutions=np.array([solutions[i] for i in range(len(solutions) ) if i!=index])
        bien_place, mal_place = check_correct(solution, word)

        if render==0:
            print(counter, solution, word, bien_place, mal_place )
    return counter

def solver_triche(word, list_mot, list_domain, render=0):
    def find_bien_place(solution, n, count_bien, list_index, word):
        """
        qui génère une dictionnary qui retourne la valeur probabiliste
        pour un mot donné comme indice.
        """
        counter=0
        for i in range(n):
            if i not in list_index:
                avant=solution[i]
                solution[i]="_"
                bien_place_new,_ =check_correct(solution, word)
                counter+=1
                if bien_place_new<count_bien:

                    solution[i]=avant
                    return i,counter
        print("Erreur : func_find_bien_place")

    def correct_mal_place(solution, n, count_bien, count_mal, list_index, word):
        """
        retourne l'indice pour un mot dans la liste
        """
        counter=0
        for i in range(n):
            if i not in list_index:
                for j in range(n):
                    if j not in list_index:
                        temp=solution[j]
                        solution[j]=solution[i]
                        counter+=1
                        bien_place_new, mal_place_new =check_correct(solution, word)
                        if bien_place_new>count_bien :
                            return j,bien_place_new,mal_place_new,counter
                        solution[j]=temp
    n=len(word)
    pb=Problem()
    solutions=deepcopy(list_mot)

    #Fixer la domaine à l'aide de list_domain
    for i in range(n):
        name_variable=str(i)
        pb.addVariables(name_variable,list_domain)

    # Générer alératoirement un mot
    index=np.random.randint(len(solutions))
    solution=solutions[index]
    solutions=np.array([ solutions[i] for i in range(solutions.shape[0] ) if i!=index])
    counter=1
    bien_fix=0
    list_index=[]

    bien_place=0
    while bien_place<n:

        bien_place, mal_place = check_correct(solution, word)
        if bien_place==0:
            if mal_place==0:
                for i in range(n):
                    for j in range(n):
                        if solution[i] in pb._variables[str(j)]:
                            pb._variables[str(j)].remove(solution[i])
                            solutions=solutions[np.where(solutions[:,j]!=solution[i])]
            else:
                for i in range(n):
                    if solution[i] in pb._variables[str(i)]:
                        pb._variables[str(i)].remove(solution[i])
                        solutions=solutions[np.where(solutions[:,i]!=solution[i])]
        if bien_place==n:
            print(counter, solution, word, bien_place, mal_place )
            break

        #chercher l'indice des lettre bien place
        while bien_place>bien_fix:
            ind,count=find_bien_place(solution, n, bien_place, list_index, word)
            list_index.append(ind)
            solutions=solutions[np.where(solutions[:,ind]==solution[ind])]
            counter+=count
            bien_fix+=1

        bien_place, mal_place = check_correct(solution, word)
        if bien_place==n:
            print(counter, solution, word, bien_place, mal_place )
            break

        #corriger le mot pour obtenir le vrai index de la lettre.
        while mal_place>0:
            j,bien_place,mal_place_new,count=correct_mal_place(solution, n, bien_place, mal_place,list_index, word)
            list_index.append(j)
            bien_fix+=1
            solutions=solutions[np.where(solutions[:,j]==solution[j])]
            counter+=count
            mal_place=mal_place_new

        # Générer alératoirement un mot
        index=np.random.randint(len(solutions))
        solution=solutions[index]
        solutions=np.array([ solutions[i] for i in range(solutions.shape[0] ) if i!=index])
        bien_place, mal_place = check_correct(solution, word)
        counter+=1
        if render==0:
            print(counter, solution, word, bien_place, mal_place )
    return counter

dictionnary=readfile('./dico.txt')
assert((check_correct("tarte","dette"))==(2,1))
assert((check_correct("bonjour","nobjour")==(5,2)))


n=4
list_mot=np.array(dictionnary[n])
list_domain=[list(set(np.array(list_mot)[:,i] )) for i in range(n) ]

for i in range(20):
    word=list_mot[np.random.randint(list_mot.shape[0])]
    solver_a2(word,list_mot,list_domain)
    print("============Word ",i+1,"Found===============")
