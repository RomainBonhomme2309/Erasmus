
city(bordeaux). 
city(rennes). 
city(paris). 
city(lyon).
city(marseille).

road(bordeaux,paris,580).
road(bordeaux,marseille,650).
road(bordeaux,lyon,550).
road(bordeaux,rennes,450).

road(rennes,paris,350).
road(rennes,marseille,1000).
road(rennes,lyon,700).

road(marseille,lyon,300).
road(marseille,paris,750).

road(paris,lyon,460).

start(bordeaux).

% Define edges (bidirectional)
edge(X, Y, D) :- road(X, Y, D).
edge(Y, X, D) :- road(X, Y, D).

% Path variables: "tour(X, Y)" is true if there's a path from X to Y
1 { tour(X, Y) : edge(X, Y, _) } 1 :- city(X), X != start.

% Ensure every city is visited exactly once
visited(X) :- tour(X, _).
visited(X) :- tour(_, X).
:- city(X), not visited(X).

% No self-loops
:- tour(X, X).

% Ensure the tour forms a single cycle starting and ending at Bordeaux
reachable(X) :- start(X).
reachable(Y) :- reachable(X), tour(X, Y).
:- city(X), not reachable(X).

% Constraints to eliminate invalid tours
:- not reachable(bordeaux).

% Calculate the total distance of the tour
distance(D) :- tour(X, Y), edge(X, Y, D).
total_distance(Total) :- Total = #sum { D : distance(D) }.

% Display the tour and the total distance
#show tour/2.
#show total_distance/1.
