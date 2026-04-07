# Modelare Teoretica si Simulare Numerica a Difuziei in 2D

## 1. Context fizic

Se considera o camera de dimensiuni 5 m x 5 m in care evolueaza trei marimi scalare:
- temperatura T(x, y, t) [degC]
- presiunea p(x, y, t) [hPa]
- umiditatea H(x, y, t) [%]

Se presupune transport dominant prin difuzie (fara termen de convectie), cu o sursa localizata in coltul camerei.

## 2. Modelul matematic (PDE)

Pentru fiecare marime u in {T, p, H}, modelul este ecuatia de difuzie:

$$
\frac{\partial u}{\partial t} = D_u \nabla^2 u
$$

unde:
- Du este coeficientul de difuzie specific marimii u
- \nabla^2 este operatorul Laplace in 2D:

$$
\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}
$$

### 2.1. Conditii initiale

La momentul initial, campul este uniform la valorile ambientale:

$$
u(x, y, 0) = u_{amb}
$$

### 2.2. Conditii la frontiera

- la sursa (coltul de injectie): conditie Dirichlet, valoare impusa

$$
u = u_{src}
$$

- pe peretii camerei: flux nul (frontiera izolata), conditie Neumann

$$
\frac{\partial u}{\partial n} = 0
$$

Aceasta inseamna ca nu exista flux net prin pereti.

## 3. Discretizare numerica (metoda Euler explicit + diferente finite)

Domeniul este discretizat cu pas spatial \Delta x = \Delta y = 0.1 m.
Timpul este discretizat cu pas \Delta t.

Notand cu u(i,j)^n aproximarea in nodul (i,j) la timpul n, schema explicita este:

$$
u_{i,j}^{n+1} = u_{i,j}^{n} + D_u \Delta t \left[
\frac{u_{i+1,j}^{n} - 2u_{i,j}^{n} + u_{i-1,j}^{n}}{\Delta x^2}
+
\frac{u_{i,j+1}^{n} - 2u_{i,j}^{n} + u_{i,j-1}^{n}}{\Delta y^2}
\right]
$$

Aceasta formula este aplicata pe nodurile interioare. Dupa fiecare pas:
- se impun conditiile de frontiera Neumann
- se reaplica sursa in zona de colt (patch local), pentru cuplare corecta cu interiorul

## 4. Stabilitate numerica

Pentru schema explicita in 2D (cu \Delta x = \Delta y), conditia CFL de stabilitate este:

$$
\Delta t \le \frac{\Delta x^2}{4 D_{max}}
$$

unde Dmax este cel mai mare coeficient de difuzie dintre marimile simulate.

In implementare se foloseste un factor de siguranta:

$$
\Delta t = 0.8 \cdot \frac{\Delta x^2}{4 D_{max}}
$$

ceea ce reduce riscul de instabilitate numerica.

## 5. Interpretare fizica a rezultatelor

Modelul produce propagarea in timp a perturbatiilor de la sursa spre restul domeniului:
- temperatura creste local in jurul sursei si se netezeste spatial
- presiunea se ajusteaza gradual fata de valoarea ambientala
- umiditatea difuzeaza progresiv spre interior

Pe termen lung, campurile tind spre un regim cvasi-stationar impus de sursa si de conditiile la frontiera.

## 6. Validare recomandata

Pentru validare, se poate compara:
- solutia numerica Euler
- solutia teoretica (profil analitic de difuzie, unde este aplicabil)

Metrici utile:
- eroare RMSE
- eroare relativa medie
- analiza evolutiei in puncte fixe din domeniu

Aceasta validare arata daca discretizarea reproduce corect dinamica modelului fizic.

## 7. Rolul Random Forest in proiect

Random Forest nu inlocuieste modelul fizic bazat pe PDE. El este un model data-driven care:
- poate aproxima rapid iesiri pe baza datelor de antrenare
- nu impune explicit legi de conservare sau conditii fizice la frontiera
- poate extrapola slab in regimuri care nu apar in date

Concluzie metodologica:
- modelul de baza pentru fizica computationala ramane modelul PDE + simulare numerica
- Random Forest este util ca benchmark sau surrogate model, complementar modelului fizic

## 8. Concluzie

Proiectul se incadreaza corect in fizica computationala prin:
- formulare de model teoretic (ecuatii diferentiale partiale)
- discretizare numerica stabila (Euler explicit)
- simulare in timp pe domeniu 2D
- comparatie critica intre abordarea fizica si una data-driven

Acest cadru ofera atat interpretabilitate fizica, cat si posibilitatea unei evaluari cantitative a performantelor numerice.
