
(define (domain action-castle)
   (:requirements :strips :typing)
   (:types player location direction monster item)

   (:action go
      :parameters (?dir - direction ?p - player ?l1 - location ?l2 - location)
      :precondition (and (at ?p ?l1) (connected ?l1 ?dir ?l2) (not (blocked ?l1 ?dir ?l2)))
      :effect (and (at ?p ?l2) (not (at ?p ?l1)))
   )

   (:action get
   :parameters (?p - player ?i - item ?l - location)
   :precondition (and (at ?p ?l) (at ?i ?l))
   :effect (and (inventory ?p ?i) (not (at ?i ?l)))
   )

   (:action drop
      :parameters (?p - player ?i - item ?l - location)
      :precondition (and (inventory ?p ?i) (at ?p ?l))
      :effect (and (not (inventory?p ?i)) (at ?i ?l))
   )

   (:action fish
      :parameters (?p - player ?l - location) 
      :precondition (and (at ?p ?l) (haslake ?l) (inventory ?p pole))
      :effect (and(inventory ?p fish)) 
   )
)
