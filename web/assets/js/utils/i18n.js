import { getLocale } from './utils.js'
const locale = getLocale()

const frFR = {
}
export const $t = (key) => {
    const fr = frFR[key]
    return locale === 'fr-FR' && fr ? fr : key
}