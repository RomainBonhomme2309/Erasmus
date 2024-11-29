
(define (problem feed-troll)

   (:objects
      npc - player
      cottage gardenpath fishingpond trollcity  - location
      in out north south east west up down - direction
      pole fish - item
      troll - monster
   )

   (:init
      (connected cottage out gardenpath)
      (connected gardenpath in cottage)
      (connected gardenpath south fishingpond)
      (connected fishingpond north gardenpath)
      (connected trollcity south gardenpath)
      (connected gardenpath north trollcity)
      (haslake fishingpond)
      (at npc cottage)
      (at pole cottage)
      (at troll trollcity)
      (hungry troll)
   )

   (:goal (and (not (hungry troll))))
   )
