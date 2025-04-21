'use client'
import { useState } from 'react'
import { Label, Listbox, ListboxButton, ListboxOption, ListboxOptions } from '@headlessui/react'
import { ChevronUpDownIcon } from '@heroicons/react/16/solid'
import { CheckIcon } from '@heroicons/react/20/solid'

  const heroes = [
      {
        id: 'abaddon',
        name: 'Abaddon',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/abaddon.png',
      },
      {
        id: 'alchemist',
        name: 'Alchemist',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/alchemist.png',
      },
      {
        id: 'ancient-apparition',
        name: 'Ancient Apparition',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/ancient_apparition.png',
      },
      {
        id: 'anti-mage',
        name: 'Anti-Mage',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/antimage.png',
      },
      {
        id: 'arc-warden',
        name: 'Arc Warden',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/arc_warden.png',
      },
      {
        id: 'axe',
        name: 'Axe',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/axe.png',
      },
      {
        id: 'bane',
        name: 'Bane',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/bane.png',
      },
      {
        id: 'batrider',
        name: 'Batrider',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/batrider.png',
      },
      {
        id: 'beastmaster',
        name: 'Beastmaster',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/beastmaster.png',
      },
      {
        id: 'bloodseeker',
        name: 'Bloodseeker',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/bloodseeker.png',
      },
      {
        id: 'bounty-hunter',
        name: 'Bounty Hunter',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/bounty_hunter.png',
      },
      {
        id: 'brewmaster',
        name: 'Brewmaster',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/brewmaster.png',
      },
      {
        id: 'bristleback',
        name: 'Bristleback',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/bristleback.png',
      },
      {
        id: 'broodmother',
        name: 'Broodmother',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/broodmother.png',
      },
      {
        id: 'centaur-warrunner',
        name: 'Centaur Warrunner',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/centaur.png',
      },
      {
        id: 'chaos-knight',
        name: 'Chaos Knight',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/chaos_knight.png',
      },
      {
        id: 'chen',
        name: 'Chen',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/chen.png',
      },
      {
        id: 'clinkz',
        name: 'Clinkz',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/clinkz.png',
      },
      {
        id: 'clockwerk',
        name: 'Clockwerk',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/rattletrap.png',
      },
      {
        id: 'crystal-maiden',
        name: 'Crystal Maiden',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/crystal_maiden.png',
      },
      {
        id: 'dark-seer',
        name: 'Dark Seer',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/dark_seer.png',
      },
      {
        id: 'dark-willow',
        name: 'Dark Willow',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/dark_willow.png',
      },
      {
        id: 'dawnbreaker',
        name: 'Dawnbreaker',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/dawnbreaker.png',
      },
      {
        id: 'dazzle',
        name: 'Dazzle',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/dazzle.png',
      },
      {
        id: 'death-prophet',
        name: 'Death Prophet',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/death_prophet.png',
      },
      {
        id: 'disruptor',
        name: 'Disruptor',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/disruptor.png',
      },
      {
        id: 'doom',
        name: 'Doom',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/doom_bringer.png',
      },
      {
        id: 'dragon-knight',
        name: 'Dragon Knight',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/dragon_knight.png',
      },
      {
        id: 'drow-ranger',
        name: 'Drow Ranger',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/drow_ranger.png',
      },
      {
        id: 'earth-spirit',
        name: 'Earth Spirit',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/earth_spirit.png',
      },
      {
        id: 'earthshaker',
        name: 'Earthshaker',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/earthshaker.png',
      },
      {
        id: 'elder-titan',
        name: 'Elder Titan',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/elder_titan.png',
      },
      {
        id: 'ember-spirit',
        name: 'Ember Spirit',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/ember_spirit.png',
      },
      {
        id: 'enchantress',
        name: 'Enchantress',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/enchantress.png',
      },
      {
        id: 'enigma',
        name: 'Enigma',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/enigma.png',
      },
      {
        id: 'faceless-void',
        name: 'Faceless Void',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/faceless_void.png',
      },
      {
        id: 'grimstroke',
        name: 'Grimstroke',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/grimstroke.png',
      },
      {
        id: 'gyrocopter',
        name: 'Gyrocopter',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/gyrocopter.png',
      },
      {
        id: 'hoodwink',
        name: 'Hoodwink',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/hoodwink.png',
      },
      {
        id: 'huskar',
        name: 'Huskar',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/huskar.png',
      },
      {
        id: 'invoker',
        name: 'Invoker',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/invoker.png',
      },
      {
        id: 'io',
        name: 'Io',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/wisp.png',
      },
      {
        id: 'jakiro',
        name: 'Jakiro',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/jakiro.png',
      },
      {
        id: 'juggernaut',
        name: 'Juggernaut',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/juggernaut.png',
      },
      {
        id: 'keeper-of-the-light',
        name: 'Keeper of the Light',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/keeper_of_the_light.png',
      },
      {
        id: 'kunkka',
        name: 'Kunkka',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/kunkka.png',
      },
      {
        id: 'legion-commander',
        name: 'Legion Commander',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/legion_commander.png',
      },
      {
        id: 'leshrac',
        name: 'Leshrac',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/leshrac.png',
      },
      {
        id: 'lich',
        name: 'Lich',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/lich.png',
      },
      {
        id: 'lifestealer',
        name: 'Lifestealer',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/life_stealer.png',
      },
      {
        id: 'lina',
        name: 'Lina',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/lina.png',
      },
      {
        id: 'lion',
        name: 'Lion',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/lion.png',
      },
      {
        id: 'lone-druid',
        name: 'Lone Druid',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/lone_druid.png',
      },
      {
        id: 'luna',
        name: 'Luna',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/luna.png',
      },
      {
        id: 'lycan',
        name: 'Lycan',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/lycan.png',
      },
      {
        id: 'magnus',
        name: 'Magnus',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/magnataur.png',
      },
      {
        id: 'marci',
        name: 'Marci',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/marci.png',
      },
      {
        id: 'mars',
        name: 'Mars',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/mars.png',
      },
      {
        id: 'medusa',
        name: 'Medusa',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/medusa.png',
      },
      {
        id: 'meepo',
        name: 'Meepo',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/meepo.png',
      },
      {
        id: 'mirana',
        name: 'Mirana',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/mirana.png',
      },
      {
        id: 'monkey-king',
        name: 'Monkey King',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/monkey_king.png',
      },
      {
        id: 'morphling',
        name: 'Morphling',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/morphling.png',
      },
      {
        id: 'muerta',
        name: 'Muerta',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/muerta.png',
      },
      {
        id: 'naga-siren',
        name: 'Naga Siren',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/naga_siren.png',
      },
      {
        id: 'natures-prophet',
        name: "Nature's Prophet",
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/furion.png',
      },
      {
        id: 'necrophos',
        name: 'Necrophos',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/necrolyte.png',
      },
      {
        id: 'night-stalker',
        name: 'Night Stalker',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/night_stalker.png',
      },
      {
        id: 'nyx-assassin',
        name: 'Nyx Assassin',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/nyx_assassin.png',
      },
      {
        id: 'ogre-magi',
        name: 'Ogre Magi',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/ogre_magi.png',
      },
      {
        id: 'omniknight',
        name: 'Omniknight',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/omniknight.png',
      },
      {
        id: 'oracle',
        name: 'Oracle',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/oracle.png',
      },
      {
        id: 'outworld-destroyer',
        name: 'Outworld Destroyer',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/obsidian_destroyer.png',
      },
      {
        id: 'pangolier',
        name: 'Pangolier',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/pangolier.png',
      },
      {
        id: 'phantom-assassin',
        name: 'Phantom Assassin',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/phantom_assassin.png',
      },
      {
        id: 'phantom-lancer',
        name: 'Phantom Lancer',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/phantom_lancer.png',
      },
      {
        id: 'phoenix',
        name: 'Phoenix',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/phoenix.png',
      },
      {
        id: 'primal-beast',
        name: 'Primal Beast',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/primal_beast.png',
      },
      {
        id: 'puck',
        name: 'Puck',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/puck.png',
      },
      {
        id: 'pudge',
        name: 'Pudge',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/pudge.png',
      },
      {
        id: 'pugna',
        name: 'Pugna',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/pugna.png',
      },
      {
        id: 'queen-of-pain',
        name: 'Queen of Pain',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/queenofpain.png',
      },
      {
        id: 'razor',
        name: 'Razor',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/razor.png',
      },
      {
        id: 'riki',
        name: 'Riki',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/riki.png',
      },
      {
        id: 'rubick',
        name: 'Rubick',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/rubick.png',
      },
      {
        id: 'sand-king',
        name: 'Sand King',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/sand_king.png',
      },
      {
        id: 'shadow-demon',
        name: 'Shadow Demon',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/shadow_demon.png',
      },
      {
        id: 'shadow-fiend',
        name: 'Shadow Fiend',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/nevermore.png',
      },
      {
        id: 'shadow-shaman',
        name: 'Shadow Shaman',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/shadow_shaman.png',
      },
      {
        id: 'silencer',
        name: 'Silencer',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/silencer.png',
      },
      {
        id: 'skywrath-mage',
        name: 'Skywrath Mage',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/skywrath_mage.png',
      },
      {
        id: 'slardar',
        name: 'Slardar',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/slardar.png',
      },
      {
        id: 'slark',
        name: 'Slark',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/slark.png',
      },
      {
        id: 'snapfire',
        name: 'Snapfire',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/snapfire.png',
      },
      {
        id: 'sniper',
        name: 'Sniper',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/sniper.png',
      },
      {
        id: 'spectre',
        name: 'Spectre',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/spectre.png',
      },
      {
        id: 'spirit-breaker',
        name: 'Spirit Breaker',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/spirit_breaker.png',
      },
      {
        id: 'storm-spirit',
        name: 'Storm Spirit',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/storm_spirit.png',
      },
      {
        id: 'sven',
        name: 'Sven',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/sven.png',
      },
      {
        id: 'techies',
        name: 'Techies',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/techies.png',
      },
      {
        id: 'templar-assassin',
        name: 'Templar Assassin',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/templar_assassin.png',
      },
      {
        id: 'terrorblade',
        name: 'Terrorblade',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/terrorblade.png',
      },
      {
        id: 'tidehunter',
        name: 'Tidehunter',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/tidehunter.png',
      },
      {
        id: 'timbersaw',
        name: 'Timbersaw',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/shredder.png',
      },
      {
        id: 'tinker',
        name: 'Tinker',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/tinker.png',
      },
      {
        id: 'tiny',
        name: 'Tiny',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/tiny.png',
      },
      {
        id: 'treant-protector',
        name: 'Treant Protector',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/treant.png',
      },
      {
        id: 'troll-warlord',
        name: 'Troll Warlord',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/troll_warlord.png',
      },
      {
        id: 'tusk',
        name: 'Tusk',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/tusk.png',
      },
      {
        id: 'underlord',
        name: 'Underlord',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/abyssal_underlord.png',
      },
      {
        id: 'undying',
        name: 'Undying',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/undying.png',
      },
      {
        id: 'ursa',
        name: 'Ursa',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/ursa.png',
      },
      {
        id: 'vengeful-spirit',
        name: 'Vengeful Spirit',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/vengefulspirit.png',
      },
      {
        id: 'venomancer',
        name: 'Venomancer',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/venomancer.png',
      },
      {
        id: 'viper',
        name: 'Viper',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/viper.png',
      },
      {
        id: 'visage',
        name: 'Visage',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/visage.png',
      },
      {
        id: 'void-spirit',
        name: 'Void Spirit',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/void_spirit.png',
      },
      {
        id: 'warlock',
        name: 'Warlock',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/warlock.png',
      },
      {
        id: 'weaver',
        name: 'Weaver',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/weaver.png',
      },
      {
        id: 'windranger',
        name: 'Windranger',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/windrunner.png',
      },
      {
        id: 'winter-wyvern',
        name: 'Winter Wyvern',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/winter_wyvern.png',
      },
      {
        id: 'witch-doctor',
        name: 'Witch Doctor',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/witch_doctor.png',
      },
      {
        id: 'wraith-king',
        name: 'Wraith King',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/skeleton_king.png',
      },
      {
        id: 'zeus',
        name: 'Zeus',
        avatar: 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/zuus.png',
      },
    ]

    // The function for dropdown box
    function Select({value, onChange }) {
      return (
        //Lists all the heroes available 
        //First section renders through all available heroes selections, second part does updating and changes the status 
        <Listbox value={value} onChange={onChange}>
          <Label className="block text-sm/6 font-medium text-white">Select a Hero</Label>
          <div className="relative mt-2">
            {/*Shows selection*/}
            <ListboxButton className="grid w-full cursor-default grid-cols-1 rounded-md bg-white py-1.5 pr-2 pl-3 text-left text-gray-900 outline-1 outline-gray-300 focus:outline-2">
              <span className="flex items-center gap-3 pr-6">
                {value ? (
                  <>
                    {/*If a hero exist, show avatar and name in the box.*/}
                    <img alt="" src={value.avatar} className="size-5 shrink-0 rounded-full" />
                    <span className="block truncate">{value.name}</span>
                  </>
                ) : (
                  <span className="text-gray-400">Choose a hero</span>
                )}
              </span>
              <ChevronUpDownIcon className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 size-5" />
            </ListboxButton>
    
            <ListboxOptions className="absolute z-10 mt-1 max-h-56 w-full overflow-auto rounded-md bg-white py-1 shadow-lg ring-1 ring-black/5 focus:outline-none opacity-">
              {/*Renders through all the heroes possible*/}
              {heroes.map((hero) => (
                <ListboxOption
                  key={hero.id}
                  value={hero}
                  className="cursor-default select-none relative py-2 pl-3 pr-9 text-gray-900 hover:bg-indigo-600 hover:text-white"
                >
                  <div className="flex items-center">
                    <img src={hero.avatar} alt="" className="size-5 shrink-0 rounded-full" />
                    <span className="ml-3 block truncate">{hero.name}</span>
                  </div>

                  {/*Icon showed if selected */}
                  {value?.id === hero.id && (
                    <span className="absolute inset-y-0 right-0 flex items-center pr-4 text-indigo-600">
                      <CheckIcon className="size-5" aria-hidden="true" />
                    </span>
                  )}
                </ListboxOption>
              ))}
            </ListboxOptions>
          </div>
        </Listbox>
      );
    }

export default Select;