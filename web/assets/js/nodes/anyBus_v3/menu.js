class menu {

	_ext = null

	constructor(extension) {
        this.ext = extension
	}

	addMenuHandler = (nodeType, cb)=> {
		const getOpts = nodeType.prototype.getExtraMenuOptions;
		nodeType.prototype.getExtraMenuOptions = function () {
			const r = getOpts.apply(this, arguments);
			cb.apply(this, arguments);
			return r;
		};
	}
	
	viewProfile = function(_, options){
		options.unshift({
			content: this.ext.$t("\ud83d\udc30 View Bus Info..."),
			callback: (value, options, e, menu, node) => {
				console.log(window.marascott[this.ext.name].nodeToSync)
			}
		})
	}

	get ext(){
        return this._ext;
    }
    
    set ext(extension){
        this._ext = extension;
    }

}

export { menu }