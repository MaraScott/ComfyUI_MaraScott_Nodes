import { $t } from './../../utils/i18n.js'

class menu {

	static addMenuHandler = (nodeType, cb)=> {
		const getOpts = nodeType.prototype.getExtraMenuOptions;
		nodeType.prototype.getExtraMenuOptions = function () {
			const r = getOpts.apply(this, arguments);
			cb.apply(this, arguments);
			return r;
		};
	}
	
	static viewProfile = function(_, options){
		options.unshift({
			content: $t("\ud83d\udc30 View Bus Info..."),
			callback: (value, options, e, menu, node) => {
				console.log(window.marascott.anyBus_v3.nodeToSync)
			}
		})
	}
}

export { menu }