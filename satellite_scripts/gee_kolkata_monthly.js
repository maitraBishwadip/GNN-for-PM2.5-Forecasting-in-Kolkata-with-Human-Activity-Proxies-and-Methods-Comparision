/* ============================================================================
 * Kolkata monthly satellite fields — VIIRS night-lights + MODIS MAIAC AOD
 * ----------------------------------------------------------------------------
 * HOW TO RUN (no install needed):
 *   1. Go to https://code.earthengine.google.com  (sign in with a Google
 *      account that has Earth Engine access — free: https://earthengine.google.com/signup ).
 *   2. Paste this whole file into the editor and click "Run".
 *   3. Open the "Tasks" tab (right panel) and click "Run" on the 2 export
 *      tasks. GeoTIFFs land in your Google Drive folder "kolkata_satellite".
 *   4. Download them and run  plot_satellite_geotiffs.py  to make the figures.
 *
 * Each output is a MULTI-BAND GeoTIFF: one band per month (2024-01 ... 2026-03).
 * ==========================================================================*/

// ---- Region of interest -----------------------------------------------------
// Default: bounding box around the 7 stations. Immediately runnable.
var kolkata = ee.Geometry.Rectangle([88.30, 22.45, 88.46, 22.65]);
// To use your exact city polygon instead, upload geodata/kolkata.geojson as a
// GEE asset (Assets > New > Shapefile/GeoJSON) and uncomment:
// var kolkata = ee.FeatureCollection('projects/your-project/assets/kolkata').geometry();

Map.centerObject(kolkata, 11);

// ---- Study period -----------------------------------------------------------
var start = ee.Date('2024-01-01');
var end   = ee.Date('2026-04-01');                       // exclusive
var nMonths = ee.Number(end.difference(start, 'month')).round();
var months  = ee.List.sequence(0, nMonths.subtract(1));

// ---- Source collections -----------------------------------------------------
// VIIRS DNB monthly stray-light-corrected night lights (avg radiance).
var viirs = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
              .select('avg_rad');
// MODIS MAIAC 1-km aerosol optical depth at 550 nm (scale factor 0.001).
var aod = ee.ImageCollection('MODIS/061/MCD19A2_GRANULES')
            .select('Optical_Depth_055');

// ---- Build a monthly multi-band stack --------------------------------------
function monthlyStack(coll, scaleFactor) {
  var imgs = months.map(function (m) {
    m = ee.Number(m);
    var s = start.advance(m, 'month');
    var e = s.advance(1, 'month');
    var name = s.format('YYYY_MM');
    return coll.filterDate(s, e).mean()
               .multiply(scaleFactor)
               .clip(kolkata)
               .rename(name);
  });
  return ee.ImageCollection(ee.List(imgs)).toBands();   // bands = months
}

var nlStack  = monthlyStack(viirs, 1);
var aodStack = monthlyStack(aod, 0.001);

// ---- Quick preview (most recent month) -------------------------------------
var lastM = ee.Number(nMonths).subtract(1);
function oneMonth(coll, m, sf) {
  var s = start.advance(m, 'month');
  return coll.filterDate(s, s.advance(1, 'month')).mean().multiply(sf).clip(kolkata);
}
Map.addLayer(oneMonth(viirs, lastM, 1),
  {min: 0, max: 60, palette: ['000000', '330066', 'cc3300', 'ffff00', 'ffffff']},
  'Night lights (latest month)');
Map.addLayer(oneMonth(aod, lastM, 0.001),
  {min: 0, max: 1.2, palette: ['ffffcc', 'fd8d3c', 'bd0026']},
  'AOD (latest month)');
print('Months exported per variable:', nMonths);

// ---- Export to Drive (2 tasks) ---------------------------------------------
Export.image.toDrive({
  image: nlStack, description: 'kolkata_nightlight_monthly',
  folder: 'kolkata_satellite', fileNamePrefix: 'kolkata_nightlight_monthly',
  region: kolkata, scale: 500, crs: 'EPSG:4326', maxPixels: 1e9
});
Export.image.toDrive({
  image: aodStack, description: 'kolkata_aod_monthly',
  folder: 'kolkata_satellite', fileNamePrefix: 'kolkata_aod_monthly',
  region: kolkata, scale: 1000, crs: 'EPSG:4326', maxPixels: 1e9
});
