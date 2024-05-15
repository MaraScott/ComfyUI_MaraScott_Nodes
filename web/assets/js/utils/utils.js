const getLocale = function(){
    const locale = localStorage['MaraScott.Locale'] || localStorage['Comfy.Settings.MaraScott.Locale'] || 'en-US'
    return locale
}

export { getLocale };