#!/usr/bin/env node

/**
 * Spotify Preview Finder - Node.js wrapper
 * Finds preview URLs for Spotify tracks using spotify-preview-finder package
 */

require('dotenv').config();
const spotifyPreviewFinder = require('spotify-preview-finder');

// Handle different environment variable names
process.env.SPOTIFY_CLIENT_ID = process.env.SPOTIFY_CLIENT_ID || process.env.CLIENT_ID;
process.env.SPOTIFY_CLIENT_SECRET = process.env.SPOTIFY_CLIENT_SECRET || process.env.CLIENT_SECRET;

// Get command line arguments
const args = process.argv.slice(2);

if (args.length < 2) {
  console.error('Usage: node spotify_preview_finder.js <track_name> <artist_name>');
  process.exit(1);
}

const trackName = args[0];
const artistName = args[1];
const limit = args[2] ? parseInt(args[2]) : 5;

async function findPreview() {
  try {
    const result = await spotifyPreviewFinder(trackName, artistName, limit);
    
    if (result.success && result.results.length > 0) {
      // Output as JSON for Python to parse
      console.log(JSON.stringify({
        success: true,
        searchQuery: result.searchQuery,
        tracks: result.results.map(song => ({
          name: song.name,
          trackId: song.trackId,
          albumName: song.albumName,
          releaseDate: song.releaseDate,
          popularity: song.popularity,
          durationMs: song.durationMs,
          spotifyUrl: song.spotifyUrl,
          previewUrls: song.previewUrls
        }))
      }));
    } else {
      console.log(JSON.stringify({
        success: false,
        error: result.error || 'No results found'
      }));
    }
  } catch (error) {
    console.log(JSON.stringify({
      success: false,
      error: error.message
    }));
  }
}

findPreview();
