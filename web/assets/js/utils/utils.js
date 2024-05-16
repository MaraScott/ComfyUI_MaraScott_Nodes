const getLocale = function(){
    const locale = localStorage['MaraScott.Locale'] || localStorage['Comfy.Settings.MaraScott.Locale'] || 'en-US'
    return locale
}

const loadJSON = async function (url) {
    let data = {
        "name": "default",
        "title": "AnyBus  - default"
    }
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        data = await response.json();
        // You can now use the data object
    } catch (error) {
        console.error('Error loading JSON:', error);
    }
    console.log(data);
    return data;
}

export { getLocale, loadJSON };