
(define (problem go-fish)
   (:domain action-castle)

   (:objects
      npc - player
      cottage gardenpath fishingpond - location
      in out north south east west up down - direction
      fish pole - item
   )

   (:init
      (connected cottage out gardenpath)
      (connected gardenpath in cottage)
      (connected gardenpath south fishingpond)
      (connected fishingpond north gardenpath)
      (haslake fishingpond)
      (at npc cottage)
      (at pole cottage)
   )

   (:goal (and (inventory npc fish)))
)
