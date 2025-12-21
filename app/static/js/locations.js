
/**
 * Mexico Geographic Data
 * Structure: { Country_Code: { State_Name: [City1, City2, ...] } }
 */

const MEXICO_DATA = {
    "Aguascalientes": ["Aguascalientes", "Jesús María", "Calvillo", "Rincón de Romos"],
    "Baja California": ["Tijuana", "Mexicali", "Ensenada", "Rosarito", "Tecate", "San Quintín"],
    "Baja California Sur": ["La Paz", "Cabo San Lucas", "San José del Cabo", "Loreto"],
    "Campeche": ["Campeche", "Ciudad del Carmen", "Champotón", "Escárcega"],
    "Coahuila": ["Saltillo", "Torreón", "Monclova", "Piedras Negras", "Acuña"],
    "Colima": ["Colima", "Manzanillo", "Tecomán", "Villa de Álvarez"],
    "Chiapas": ["Tuxtla Gutiérrez", "Tapachula", "San Cristóbal de las Casas", "Comitán"],
    "Chihuahua": ["Chihuahua", "Ciudad Juárez", "Delicias", "Cuauhtémoc", "Parral"],
    "Ciudad de México": ["Álvaro Obregón", "Azcapotzalco", "Benito Juárez", "Coyoacán", "Cuajimalpa", "Cuauhtémoc", "Gustavo A. Madero", "Iztacalco", "Iztapalapa", "Magdalena Contreras", "Miguel Hidalgo", "Milpa Alta", "Tláhuac", "Tlalpan", "Venustiano Carranza", "Xochimilco"],
    "Durango": ["Durango", "Gómez Palacio", "Lerdo", "Santiago Papasquiaro"],
    "Guanajuato": ["León", "Irapuato", "Celaya", "Salamanca", "Guanajuato", "San Miguel de Allende"],
    "Guerrero": ["Acapulco", "Chilpancingo", "Iguala", "Taxco", "Zihuatanejo"],
    "Hidalgo": ["Pachuca", "Tulancingo", "Tula", "Tizayuca", "Huejutla"],
    "Jalisco": ["Guadalajara", "Zapopan", "Tlaquepaque", "Tonalá", "Puerto Vallarta", "Tlajomulco", "Lagos de Moreno", "Tepatitlán"],
    "México": ["Toluca", "Ecatepec", "Naucalpan", "Tlalnepantla", "Nezahualcóyotl", "Cuautitlán Izcalli", "Chalco", "Chimalhuacán"],
    "Michoacán": ["Morelia", "Uruapan", "Zamora", "Lázaro Cárdenas", "Apatzingán"],
    "Morelos": ["Cuernavaca", "Jiutepec", "Cuautla", "Temixco"],
    "Nayarit": ["Tepic", "Bahía de Banderas", "Xalisco", "Compostela"],
    "Nuevo León": ["Monterrey", "San Pedro Garza García", "San Nicolás de los Garza", "Guadalupe", "Apodaca", "Santa Catarina", "Escobedo"],
    "Oaxaca": ["Oaxaca de Juárez", "Huatulco", "Puerto Escondido", "Salina Cruz", "Juchitán"],
    "Puebla": ["Puebla", "Tehuacán", "Cholula", "Atlixco", "San Martín Texmelucan"],
    "Querétaro": ["Querétaro", "San Juan del Río", "Corregidora", "El Marqués"],
    "Quintana Roo": ["Cancún", "Playa del Carmen", "Chetumal", "Cozumel", "Tulum"],
    "San Luis Potosí": ["San Luis Potosí", "Soledad de Graciano Sánchez", "Ciudad Valles", "Matehuala"],
    "Sinaloa": ["Culiacán", "Mazatlán", "Los Mochis", "Guasave", "Guamúchil"],
    "Sonora": ["Hermosillo", "Ciudad Obregón", "Nogales", "San Luis Río Colorado", "Guaymas", "Navojoa"],
    "Tabasco": ["Villahermosa", "Cárdenas", "Comalcalco", "Macuspana"],
    "Tamaulipas": ["Reynosa", "Matamoros", "Nuevo Laredo", "Tampico", "Ciudad Victoria", "Madero"],
    "Tlaxcala": ["Tlaxcala", "Apizaco", "Huamantla", "Chiautempan"],
    "Veracruz": ["Veracruz", "Xalapa", "Coatzacoalcos", "Córdoba", "Poza Rica", "Orizaba", "Minatitlán"],
    "Yucatán": ["Mérida", "Valladolid", "Tizimín", "Progreso", "Kanasín"],
    "Zacatecas": ["Zacatecas", "Guadalupe", "Fresnillo", "Jerez"]
};

const COUNTRIES = [
    { name: "México", code: "+52", states: MEXICO_DATA },
    { name: "Estados Unidos", code: "+1", states: {} }, // Placeholder for now
    { name: "Canadá", code: "+1", states: {} },
    { name: "Colombia", code: "+57", states: {} },
    { name: "España", code: "+34", states: {} },
    { name: "Argentina", code: "+54", states: {} }
];

// Helper functions to populate dropdowns
function populateCountries(selectId) {
    const select = document.getElementById(selectId);
    if (!select) return;
    select.innerHTML = '';
    COUNTRIES.forEach(c => {
        const option = document.createElement('option');
        option.value = c.name;
        option.text = c.name;
        option.dataset.code = c.code; // Store phone code
        if (c.name === 'México') option.selected = true;
        select.appendChild(option);
    });
    // Trigger state load for default
    const stateSelectId = select.dataset.targetState;
    if (stateSelectId) loadStatesForCountry(select.value, stateSelectId);
}

function loadStatesForCountry(countryName, stateSelectId) {
    const stateSelect = document.getElementById(stateSelectId);
    if (!stateSelect) return;

    stateSelect.innerHTML = '<option value="">Selecciona un Estado</option>';
    stateSelect.disabled = true;

    const country = COUNTRIES.find(c => c.name === countryName);
    if (country && Object.keys(country.states).length > 0) {
        stateSelect.disabled = false;
        Object.keys(country.states).sort().forEach(state => {
            const option = document.createElement('option');
            option.value = state;
            option.text = state;
            stateSelect.appendChild(option);
        });
    } else {
        // Fallback for countries without data -> maybe turn into text input or just "Other"
        const option = document.createElement('option');
        option.value = "Otro";
        option.text = "Otro / No listado";
        stateSelect.appendChild(option);
        stateSelect.disabled = false;
    }

    // Reset City
    const citySelectId = stateSelect.dataset.targetCity;
    if (citySelectId) {
        const citySelect = document.getElementById(citySelectId);
        citySelect.innerHTML = '<option value="">Selecciona una Ciudad</option>';
        citySelect.disabled = true;
    }
}

function loadCitiesForState(countryName, stateName, citySelectId) {
    const citySelect = document.getElementById(citySelectId);
    if (!citySelect) return;

    citySelect.innerHTML = '<option value="">Selecciona una Ciudad</option>';
    citySelect.disabled = true;

    const country = COUNTRIES.find(c => c.name === countryName);
    if (country && country.states[stateName]) {
        citySelect.disabled = false;
        country.states[stateName].sort().forEach(city => {
            const option = document.createElement('option');
            option.value = city;
            option.text = city;
            citySelect.appendChild(option);
        });
    }
}

// Function to populate Phone Code Select
function populatePhoneCodes(selectId) {
    const select = document.getElementById(selectId);
    if (!select) return;
    select.innerHTML = '';
    COUNTRIES.forEach(c => {
        const option = document.createElement('option');
        option.value = c.code; // Value is the prefix (+52)
        option.text = `${c.name} (${c.code})`;
        if (c.name === 'México') option.selected = true;
        select.appendChild(option);
    });
}
