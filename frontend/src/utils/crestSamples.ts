export interface CrestSample {
  id: string;
  name: string;
  country: string;
  league: string;
  imageUrl: string;
}

/**
 * Muestras precargadas de escudos de fútbol para la subfase 1.2.
 * Se pueden reemplazar por datos reales del backend en posteriores subfases.
 */
export const CREST_SAMPLES: CrestSample[] = [
  {
    id: 'boca',
    name: 'Boca Juniors',
    country: 'Argentina',
    league: 'Primera División',
    imageUrl:
      'https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Estadio_Lnier_Dellacasa_dela_FC_Barcelona_2015.jpg/320px-Estadio_Lnier_Dellacasa_dela_FC_Barcelona_2015.jpg',
  },
  {
    id: 'river',
    name: 'River Plate',
    country: 'Argentina',
    league: 'Primera División',
    imageUrl:
      'https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Estadio_Lnier_Dellacasa_dela_FC_Barcelona_2015.jpg/320px-Estadio_Lnier_Dellacasa_dela_FC_Barcelona_2015.jpg',
  },
  {
    id: 'barcelona',
    name: 'FC Barcelona',
    country: 'España',
    league: 'La Liga',
    imageUrl:
      'https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Estadio_Lnier_Dellacasa_dela_FC_Barcelona_2015.jpg/320px-Estadio_Lnier_Dellacasa_dela_FC_Barcelona_2015.jpg',
  },
  {
    id: 'real-madrid',
    name: 'Real Madrid',
    country: 'España',
    league: 'La Liga',
    imageUrl:
      'https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Estadio_Lnier_Dellacasa_dela_FC_Barcelona_2015.jpg/320px-Estadio_Lnier_Dellacasa_dela_FC_Barcelona_2015.jpg',
  },
  {
    id: 'juventus',
    name: 'Juventus',
    country: 'Italia',
    league: 'Serie A',
    imageUrl:
      'https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Estadio_Lnier_Dellacasa_dela_FC_Barcelona_2015.jpg/320px-Estadio_Lnier_Dellacasa_dela_FC_Barcelona_2015.jpg',
  },
];
